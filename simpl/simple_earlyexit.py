import time
from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn

from utils.utils import gpu, init_weights
from .simpl import ActorNet, LaneNet, MLPDecoder, SymmetricFusionTransformer


class FusionNet(nn.Module):
    def __init__(self, device, config):
        super().__init__()
        self.device = device

        d_embed = config["d_embed"]
        self.exit_layer = min(config.get("exit_layer", 1), config["n_scene_layer"] - 1)

        self.proj_actor = nn.Sequential(
            nn.Linear(config["d_actor"], d_embed),
            nn.LayerNorm(d_embed),
            nn.ReLU(inplace=True),
        )
        self.proj_lane = nn.Sequential(
            nn.Linear(config["d_lane"], d_embed),
            nn.LayerNorm(d_embed),
            nn.ReLU(inplace=True),
        )
        self.proj_rpe_scene = nn.Sequential(
            nn.Linear(config["d_rpe_in"], config["d_rpe"]),
            nn.LayerNorm(config["d_rpe"]),
            nn.ReLU(inplace=True),
        )

        self.fuse_scene = SymmetricFusionTransformer(
            self.device,
            d_model=d_embed,
            d_edge=config["d_rpe"],
            n_head=config["n_scene_head"],
            n_layer=config["n_scene_layer"],
            dropout=config["dropout"],
            update_edge=config["update_edge"],
        )

    def _run_remaining_layers(self, scene_states: List[Dict[str, Tensor]]) -> Tensor:
        actors_final = []
        for scene_state in scene_states:
            tokens = scene_state["tokens"]
            edge = scene_state["edge"]
            scene_mask = scene_state["scene_mask"]
            actor_count = scene_state["actor_count"]

            for mod in self.fuse_scene.fusion[self.exit_layer + 1 :]:
                tokens, edge, _ = mod(tokens, edge, scene_mask)

            actors_final.append(tokens[:actor_count])

        return torch.cat(actors_final, dim=0)

    def forward(
        self,
        actors: Tensor,
        actor_idcs: List[Tensor],
        lanes: Tensor,
        lane_idcs: List[Tensor],
        rpe_prep: Dict[str, Tensor],
        stop_at_exit: bool = False,
        capture_exit_states: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, List[Dict[str, Tensor]]]:
        actors = self.proj_actor(actors)
        lanes = self.proj_lane(lanes)

        actors_final, lanes_final, actors_exit, exit_scene_tokens = [], [], [], []
        exit_states = []
        for a_idcs, l_idcs, rpes in zip(actor_idcs, lane_idcs, rpe_prep):
            scene_actors = actors[a_idcs]
            scene_lanes = lanes[l_idcs]
            tokens = torch.cat([scene_actors, scene_lanes], dim=0)
            edge = self.proj_rpe_scene(rpes["scene"].permute(1, 2, 0))

            exit_tokens = None
            for layer_idx, mod in enumerate(self.fuse_scene.fusion):
                tokens, edge, _ = mod(tokens, edge, rpes["scene_mask"])
                if layer_idx == self.exit_layer:
                    exit_tokens = tokens[: len(a_idcs)]
                    if capture_exit_states:
                        exit_states.append(
                            {
                                "tokens": tokens,
                                "edge": edge,
                                "scene_mask": rpes["scene_mask"],
                                "actor_count": len(a_idcs),
                            }
                        )
                    if stop_at_exit:
                        break

            if exit_tokens is None:
                exit_tokens = tokens[: len(a_idcs)]

            actors_exit.append(exit_tokens)
            exit_scene_tokens.append(exit_tokens[0])
            if not stop_at_exit:
                actors_final.append(tokens[: len(a_idcs)])
                lanes_final.append(tokens[len(a_idcs) :])

        actors_exit = torch.cat(actors_exit, dim=0)
        exit_scene_tokens = torch.stack(exit_scene_tokens, dim=0)
        if stop_at_exit:
            return None, None, actors_exit, exit_scene_tokens, exit_states

        return (
            torch.cat(actors_final, dim=0),
            torch.cat(lanes_final, dim=0),
            actors_exit,
            exit_scene_tokens,
            exit_states,
        )

    def continue_from_exit(self, exit_states: List[Dict[str, Tensor]]) -> Tensor:
        return self._run_remaining_layers(exit_states)


class Simpl(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.device = device
        self.eval_output = cfg.get("eval_output", "main")
        self.selective_threshold = cfg.get("selective_threshold", 0.5)
        self.selective_score = cfg.get("selective_score", "top1_prob")

        self.actor_net = ActorNet(
            n_in=cfg["in_actor"],
            hidden_size=cfg["d_actor"],
            n_fpn_scale=cfg["n_fpn_scale"],
        )
        self.lane_net = LaneNet(
            device=self.device,
            in_size=cfg["in_lane"],
            hidden_size=cfg["d_lane"],
            dropout=cfg["dropout"],
        )
        self.fusion_net = FusionNet(device=self.device, config=cfg)
        self.pred_net = MLPDecoder(device=self.device, config=cfg)
        self.exit_pred_net = MLPDecoder(device=self.device, config=cfg)
        self.policy_head = nn.Sequential(
            nn.Linear(cfg["d_embed"], cfg["d_embed"]),
            nn.ReLU(inplace=True),
            nn.Linear(cfg["d_embed"], 1),
        )

        if cfg["init_weights"]:
            self.apply(init_weights)

    def _sync_for_timing(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _scene_scores(self, branch_out):
        scores = []
        for probs in branch_out[0]:
            target_probs = probs[0]
            if self.selective_score == "top1_prob":
                score = target_probs.max()
            elif self.selective_score == "margin":
                topk = torch.topk(target_probs, k=min(2, target_probs.shape[0])).values
                if topk.shape[0] == 1:
                    score = topk[0]
                else:
                    score = topk[0] - topk[1]
            else:
                raise ValueError(f"Unknown selective_score '{self.selective_score}'")
            scores.append(score)
        return torch.stack(scores, dim=0)

    def _subset_actor_idcs(self, actor_idcs: List[Tensor], keep_mask: Tensor) -> List[Tensor]:
        subset_idcs = []
        offset = 0
        for i, keep in enumerate(keep_mask.tolist()):
            if not keep:
                continue
            actor_count = actor_idcs[i].shape[0]
            subset_idcs.append(torch.arange(actor_count, device=self.device) + offset)
            offset += actor_count
        return subset_idcs

    def _merge_outputs(self, exit_out, main_out, continue_mask: Tensor):
        exit_cls, exit_reg, exit_aux = exit_out
        main_cls, main_reg, main_aux = main_out

        merged_cls, merged_reg, merged_aux = [], [], []
        main_idx = 0
        for scene_idx, continue_scene in enumerate(continue_mask.tolist()):
            if continue_scene:
                merged_cls.append(main_cls[main_idx])
                merged_reg.append(main_reg[main_idx])
                merged_aux.append(main_aux[main_idx])
                main_idx += 1
            else:
                merged_cls.append(exit_cls[scene_idx])
                merged_reg.append(exit_reg[scene_idx])
                merged_aux.append(exit_aux[scene_idx])

        return merged_cls, merged_reg, merged_aux

    def forward(self, data):
        actors, actor_idcs, lanes, lane_idcs, rpe = data

        actors = self.actor_net(actors)
        lanes = self.lane_net(lanes)

        if (not self.training) and self.eval_output == "selective":
            self._sync_for_timing()
            exit_start = time.perf_counter()
            _, _, actors_exit, exit_scene_tokens, exit_states = self.fusion_net(
                actors, actor_idcs, lanes, lane_idcs, rpe, stop_at_exit=True, capture_exit_states=True
            )
            exit_out = self.exit_pred_net(actors_exit, actor_idcs)
            scores = self._scene_scores(exit_out)
            exit_mask = scores >= self.selective_threshold
            continue_mask = ~exit_mask
            policy_logits = self.policy_head(exit_scene_tokens).squeeze(-1)
            self._sync_for_timing()
            exit_elapsed_ms = (time.perf_counter() - exit_start) * 1000.0

            if continue_mask.any():
                self._sync_for_timing()
                main_start = time.perf_counter()
                subset_states = [exit_states[i] for i, keep in enumerate(continue_mask.tolist()) if keep]
                subset_actor_idcs = self._subset_actor_idcs(actor_idcs, continue_mask)
                actors_final = self.fusion_net.continue_from_exit(subset_states)
                main_subset_out = self.pred_net(actors_final, subset_actor_idcs)
                selected_out = self._merge_outputs(exit_out, main_subset_out, continue_mask)
                self._sync_for_timing()
                main_elapsed_ms = (time.perf_counter() - main_start) * 1000.0
            else:
                selected_out = exit_out
                main_elapsed_ms = 0.0

            batch_size = int(exit_mask.numel())
            exited_count = int(exit_mask.sum().item())
            continued_count = batch_size - exited_count
            exit_ms_per_scene = exit_elapsed_ms / batch_size if batch_size else 0.0
            full_ms_per_scene = (
                exit_ms_per_scene + (main_elapsed_ms / continued_count)
                if continued_count
                else 0.0
            )
            return {
                "selective": selected_out,
                "exit": exit_out,
                "policy_logits": policy_logits,
                "selective_scores": scores,
                "selective_exit_mask": exit_mask,
                "selective_timing": {
                    "total_scenes": batch_size,
                    "early_exit_scenes": exited_count,
                    "full_pass_scenes": continued_count,
                    "exit_ms_per_scene": exit_ms_per_scene,
                    "full_ms_per_scene": full_ms_per_scene,
                },
            }

        if (not self.training) and self.eval_output == "exit":
            _, _, actors_exit, exit_scene_tokens, _ = self.fusion_net(
                actors, actor_idcs, lanes, lane_idcs, rpe, stop_at_exit=True
            )
            exit_out = self.exit_pred_net(actors_exit, actor_idcs)
            policy_logits = self.policy_head(exit_scene_tokens).squeeze(-1)
            return {"exit": exit_out, "policy_logits": policy_logits}

        actors_final, _, actors_exit, exit_scene_tokens, _ = self.fusion_net(
            actors, actor_idcs, lanes, lane_idcs, rpe
        )

        main_out = self.pred_net(actors_final, actor_idcs)
        exit_out = self.exit_pred_net(actors_exit, actor_idcs)
        policy_logits = self.policy_head(exit_scene_tokens).squeeze(-1)

        return {"main": main_out, "exit": exit_out, "policy_logits": policy_logits}

    def pre_process(self, data):
        actors = gpu(data["ACTORS"], self.device)
        actor_idcs = gpu(data["ACTOR_IDCS"], self.device)
        lanes = gpu(data["LANES"], self.device)
        lane_idcs = gpu(data["LANE_IDCS"], self.device)
        rpe = gpu(data["RPE"], self.device)

        return actors, actor_idcs, lanes, lane_idcs, rpe

    def _select_output(self, out, branch=None):
        if not isinstance(out, dict):
            return out

        selected_branch = self.eval_output if branch is None else branch
        if selected_branch not in out:
            raise KeyError(f"Unknown output branch '{selected_branch}'")
        return out[selected_branch]

    def post_process(self, out):
        selected = self._select_output(out)
        res_cls = selected[0]
        res_reg = selected[1]

        reg = torch.stack([trajs[0] for trajs in res_reg], dim=0)
        cls = torch.stack([probs[0] for probs in res_cls], dim=0)

        post_out = {
            "out_raw": selected,
            "traj_pred": reg,
            "prob_pred": cls,
        }

        if isinstance(out, dict):
            if "main" in out:
                post_out["main_raw"] = out["main"]
            if "exit" in out:
                post_out["exit_raw"] = out["exit"]
            if "selective" in out:
                post_out["selective_raw"] = out["selective"]
            if "policy_logits" in out:
                post_out["policy_logits"] = out["policy_logits"]
                post_out["policy_scores"] = torch.sigmoid(out["policy_logits"])
            if "selective_scores" in out:
                post_out["selective_scores"] = out["selective_scores"]
            if "selective_exit_mask" in out:
                post_out["selective_exit_mask"] = out["selective_exit_mask"]
                post_out["selective_exit_rate"] = out["selective_exit_mask"].float().mean()
            if "selective_timing" in out:
                post_out["selective_timing"] = out["selective_timing"]
            post_out["selected_branch"] = self.eval_output

        return post_out
