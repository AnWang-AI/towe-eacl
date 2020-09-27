export PYTHONPATH=./

#### 提示：如果切换bert和w2v的模式时候，需要提前删掉数据集下面的processed文件夹！


## 在服务器上运行
CUDA_VISIBLE_DEVICES=1 python src/model/trainer.py \
--data_path /data/an_wang/towe-eacl/data/14res \
--save_model_name models/Tag-LSTM.ckpt \
--use_bert \
--model Target_BiLSTM_with_bert \
--loss FacalLoss \
--epoch 60 --train_batch_size 16 --val_batch_size 16 \
--cuda

# LSTM
CUDA_VISIBLE_DEVICES=1 python src/model/trainer.py \
--data_path ./data/14res \
--save_model_name models/Model_Tag_BiLSTM.ckpt \
--model Tag_BiLSTM \
--loss CrossEntropy \
--epoch 20 --train_batch_size 32 --val_batch_size 32 \
--eval_frequency 1 \
--cuda

# graph + LSTM
CUDA_VISIBLE_DEVICES=7 python src/model/trainer.py \
--config_path ./src/model/conf_w2v_gnn_lstm.ini \
--data_path ./data/16res \
--epoch 40 --train_batch_size 32 \
--eval_frequency 2 \
--save_model_name models/Model_ExtractionNet__with_graph.ckpt_0

# bert graph LSTM
CUDA_VISIBLE_DEVICES=7 python src/model/trainer.py \
--config_path ./src/model/conf_bert_gnn_lstm.ini \
--data_path ./data/16res \
--epoch 40 --train_batch_size 32 \
--eval_frequency 2 \
--save_model_name models/Model_ExtractionNet_with_bert_with_graph.ckpt_3


## 在windows里的ipython中运行
# bert LSTM
%run src/model/trainer.py \
--data_path ./data/14res \
--save_model_name models/Model_Target_BiLSTM_with_bert.ckpt \
--use_bert \
--model Target_BiLSTM_with_bert \
--loss FacalLoss \
--epoch 40 --train_batch_size 32 --val_batch_size 32 \
--eval_frequency 2 \
--cuda

# LSTM
%run src/model/trainer.py \
--data_path ./data/14res \
--save_model_name models/Model_Tag_BiLSTM.ckpt \
--model Tag_BiLSTM \
--loss CrossEntropy \
--epoch 20 --train_batch_size 32 --val_batch_size 32 \
--eval_frequency 1 \
--cuda

%run src/model/trainer_crf.py \
--data_path ./data/14res \
--save_model_name models/Model_BiLSTM_CRF.ckpt \
--epoch 20 --train_batch_size 1 --val_batch_size 1 \
--eval_frequency 1 \
--cuda

# graph
%run src/model/trainer.py \
--data_path ./data/14res \
--save_model_name models/Model_ExtractionNet_with_graph.ckpt \
--build_graph \
--model ExtractionNet \
--loss CrossEntropy \
--epoch 40 --train_batch_size 32 --val_batch_size 32 \
--eval_frequency 2 \
--cuda

# bert + graph
%run src/model/trainer.py \
--data_path ./data/14res \
--save_model_name models/Model_ExtractionNet_with_bert_with_graph.ckpt \
--use_bert \
--build_graph \
--model Target_BiLSTM_with_bert \
--loss CrossEntropy \
--epoch 40 --train_batch_size 32 --val_batch_size 32 \
--eval_frequency 2 \
--cuda

