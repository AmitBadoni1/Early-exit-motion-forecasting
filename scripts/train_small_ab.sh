CUDA_VISIBLE_DEVICES=0 python /storage/ice-shared/cs7643/group-a/Early-exits-in-motion-forecasting/train.py \
    --features_dir data_av2/features_small/ \
    --train_batch_size 4 \
    --val_batch_size 4 \
    --val_interval 2 \
    --train_epoches 10 \
    --data_aug \
    --use_cuda \
    --logger_writer \
    --adv_cfg_path config.simpl_av2_cfg
