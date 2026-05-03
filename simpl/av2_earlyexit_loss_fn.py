import torch
import torch.nn as nn
import torch.nn.functional as F

from .av2_loss_fn import LossFunc as BaseLossFunc


class LossFunc(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device

        self.main_loss_fn = BaseLossFunc(config, device)
        self.exit_loss_fn = BaseLossFunc(config, device)

        self.exit_loss_coef = config.get("exit_loss_coef", 0.5)
        self.distill_cls_coef = config.get("distill_cls_coef", 0.0)
        self.distill_reg_coef = config.get("distill_reg_coef", 0.0)
        self.latency_gap_coef = config.get("latency_gap_coef", 0.0)
        self.latency_savings = config.get("latency_savings", 0.0)
        self.policy_loss_coef = config.get("policy_loss_coef", 0.0)
        self.policy_target_mode = config.get("policy_target_mode", "relative_fde")
        self.policy_margin = config.get("policy_margin", 0.5)
        self.policy_fde_thres = config.get("policy_fde_thres", 8.0)
        self.reg_distill = nn.SmoothL1Loss(reduction="mean")

    def forward(self, out, data):
        if not isinstance(out, dict):
            return self.main_loss_fn(out, data)

        if "main" not in out:
            return self.exit_loss_fn(out["exit"], data)
        if "exit" not in out:
            return self.main_loss_fn(out["main"], data)

        main_loss = self.main_loss_fn(out["main"], data)
        exit_loss = self.exit_loss_fn(out["exit"], data)

        loss_out = {
            "main_cls_loss": main_loss["cls_loss"],
            "main_reg_loss": main_loss["reg_loss"],
            "exit_cls_loss": exit_loss["cls_loss"],
            "exit_reg_loss": exit_loss["reg_loss"],
        }

        total_loss = main_loss["loss"] + self.exit_loss_coef * exit_loss["loss"]

        if "yaw_loss" in main_loss:
            loss_out["main_yaw_loss"] = main_loss["yaw_loss"]
        if "yaw_loss" in exit_loss:
            loss_out["exit_yaw_loss"] = exit_loss["yaw_loss"]

        distill_cls = torch.tensor(0.0, device=self.device)
        distill_reg = torch.tensor(0.0, device=self.device)

        if self.distill_cls_coef > 0.0:
            distill_cls = self._distill_cls(out["exit"][0], out["main"][0])
            total_loss = total_loss + self.distill_cls_coef * distill_cls
            loss_out["distill_cls_loss"] = distill_cls

        if self.distill_reg_coef > 0.0:
            distill_reg = self._distill_reg(out["exit"][1], out["main"][1])
            total_loss = total_loss + self.distill_reg_coef * distill_reg
            loss_out["distill_reg_loss"] = distill_reg

        if self.latency_gap_coef > 0.0:
            latency_gap = torch.relu(exit_loss["loss"] - main_loss["loss"].detach())
            latency_penalty = self.latency_savings * latency_gap
            total_loss = total_loss + self.latency_gap_coef * latency_penalty
            loss_out["latency_gap_loss"] = latency_gap
            loss_out["latency_penalty_loss"] = latency_penalty

        if self.policy_loss_coef > 0.0 and "policy_logits" in out:
            policy_targets = self._policy_targets(out["exit"], out["main"], data)
            policy_loss = F.binary_cross_entropy_with_logits(out["policy_logits"], policy_targets)
            total_loss = total_loss + self.policy_loss_coef * policy_loss
            loss_out["policy_loss"] = policy_loss
            loss_out["policy_target_mean"] = policy_targets.mean()
            loss_out["policy_prob_mean"] = torch.sigmoid(out["policy_logits"]).mean()

        loss_out["loss"] = total_loss
        return loss_out

    def _distill_cls(self, exit_cls, main_cls):
        losses = []
        for exit_probs, main_probs in zip(exit_cls, main_cls):
            losses.append(
                F.kl_div(
                    (exit_probs.clamp_min(1e-8)).log(),
                    main_probs.detach().clamp_min(1e-8),
                    reduction="batchmean",
                )
            )
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=self.device)

    def _distill_reg(self, exit_reg, main_reg):
        losses = []
        for exit_traj, main_traj in zip(exit_reg, main_reg):
            losses.append(self.reg_distill(exit_traj, main_traj.detach()))
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=self.device)

    def _policy_targets(self, exit_out, main_out, data):
        exit_reg = exit_out[1]
        main_reg = main_out[1]

        traj_fut = [x["TRAJS_POS_FUT"][0] for x in data["TRAJS"]]
        pad_fut = [x["PAD_FUT"][0].bool() for x in data["TRAJS"]]

        targets = []
        for exit_scene, main_scene, gt_scene, pad_scene in zip(exit_reg, main_reg, traj_fut, pad_fut):
            gt_scene = gt_scene.to(self.device)
            pad_scene = pad_scene.to(self.device)
            last_idx = torch.nonzero(pad_scene, as_tuple=False)[-1, 0]

            gt_final = gt_scene[last_idx]
            exit_final = exit_scene[0, :, last_idx, :2]
            main_final = main_scene[0, :, last_idx, :2]

            exit_best_fde = torch.norm(exit_final - gt_final.unsqueeze(0), dim=-1).min()
            main_best_fde = torch.norm(main_final - gt_final.unsqueeze(0), dim=-1).min()

            if self.policy_target_mode == "relative_fde":
                target = exit_best_fde <= (main_best_fde + self.policy_margin)
            elif self.policy_target_mode == "abs_fde":
                target = exit_best_fde <= self.policy_fde_thres
            elif self.policy_target_mode == "hybrid_fde":
                target = (exit_best_fde <= self.policy_fde_thres) & (
                    exit_best_fde <= (main_best_fde + self.policy_margin)
                )
            else:
                raise ValueError(f"Unknown policy_target_mode: {self.policy_target_mode}")

            targets.append(target.float())

        return torch.stack(targets, dim=0)
