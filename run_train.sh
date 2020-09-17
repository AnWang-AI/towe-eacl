export PYTHONPATH=./

CUDA_VISIBLE_DEVICES=1 python src/model/trainer.py \
--data_path /data/an_wang/towe-eacl/data/14res \
--save_model_name models/Tag-LSTM.ckpt \
--epoch 60 --train_batch_size 16 --val_batch_size 16 \
--cuda
