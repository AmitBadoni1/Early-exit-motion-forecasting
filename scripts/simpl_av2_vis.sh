CUDA_VISIBLE_DEVICES=0 python visualize.py \
  --features_dir data_av2/features/ \
  --use_cuda \
  --mode val \
  --model_path saved_models/20260429-191759_Simpl_best.tar \
  --adv_cfg_path config.vis_main \
  --visualizer simpl.av2_visualizer:Visualizer \
  --seq_id 1