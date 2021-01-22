export PYTHONPATH=./
export model=w2v_gnn_lstm
export dataset=14res

#if [ -d "./data/$dataset/processed" ]; then
#  rm -r ./data/$dataset/processed
#  echo "remove the dir ./data/$dataset/processed"
#fi

export num_mid_layers=6

CUDA_VISIBLE_DEVICES=0 python main.py \
--config_path ./src/model/config/conf_$model.ini \
--data_path ./data/$dataset \
--epoch 50 --train_batch_size 16 \
--num_mid_layers $num_mid_layers \
--eval_frequency 2 \
--save_model_name models/Model_ExtractionNet_$model_mid_layer_num_$num_mid_layers_$dataset.ckpt