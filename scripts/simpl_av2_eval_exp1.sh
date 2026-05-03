CUDA_VISIBLE_DEVICES=0 python evaluation_exp1.py \
  --features_dir data_av2/features/ \
  --train_batch_size 16 \
  --val_batch_size 16 \
  --use_cuda \
  --adv_cfg_path config.exp1_eval_cfg \
  --model_path saved_models/20260429-191759_Simpl_best.tar