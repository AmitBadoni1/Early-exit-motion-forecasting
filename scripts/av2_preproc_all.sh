cd /storage/ice-shared/cs7643/group-a/Early-exits-in-motion-forecasting || exit 1

echo "-- Processing AV2 val set..."
python data_av2/run_preprocess.py --mode val \
  --data_dir ./argoverse2_data/val/ \
  --save_dir data_av2/features/

echo "-- Processing AV2 train set..."
python data_av2/run_preprocess.py --mode train \
  --data_dir ./argoverse2_data/train/ \
  --save_dir data_av2/features/
