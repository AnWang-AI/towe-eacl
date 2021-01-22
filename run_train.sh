export PYTHONPATH=./

#### 提示：如果切换bert和w2v的模式时候，需要提前删掉数据集下面的processed文件夹！


## 在服务器上运行

# LSTM
CUDA_VISIBLE_DEVICES=0 python src/model/trainer.py \
--config_path ./src/model/config/conf_lstm.ini \
--data_path ./data/14res \
--save_model_name models/Model_Tag_BiLSTM.ckpt \
--epoch 20 --train_batch_size 32 \
--eval_frequency 2 \

# bert LSTM
CUDA_VISIBLE_DEVICES=0 python src/model/trainer.py \
--config_path ./src/model/config/conf_bert_lstm.ini \
--data_path ./data/14lap \
--epoch 40 --train_batch_size 32 \
--eval_frequency 2 \
--save_model_name models/Model_Tag_BiLSTM_bert_14lap.ckpt

# graph + LSTM
CUDA_VISIBLE_DEVICES=0 python src/model/trainer.py \
--config_path ./src/model/config/conf_w2v_gnn_lstm.ini \
--data_path ./data/16res \
--epoch 50 --train_batch_size 16 \
--num_mid_layers 4 \
--eval_frequency 2 \
--save_model_name models/Model_ExtractionNet__with_graph_16res.ckpt

# bert graph LSTM
CUDA_VISIBLE_DEVICES=0 python src/model/trainer.py \
--config_path ./src/model/config/conf_bert_gnn_lstm.ini \
--data_path ./data/14res \
--epoch 60 --train_batch_size 16 \
--eval_frequency 2 \
--save_model_name models/Model_ExtractionNet_with_bert_with_graph_14res.ckpt

