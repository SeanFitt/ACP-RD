export TASK_NAME=rumor_detection
export DATASET_NAME=pheme
export CUDA_VISIBLE_DEVICES=0

bs=32
lr=5e-5
dropout=0.1
epoch=3

python3 run.py \
  --model_name_or_path pre_trained_model/bert-base-uncased \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --do_train \
  --max_seq_length 128 \
  --per_device_train_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs $epoch \
  --output_dir checkpoints/$DATASET_NAME-bert/ \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed 32 \
  --save_strategy no \
  --evaluation_strategy epoch \
  --do_active_learning \
  --active_budget 500\
  --per_active_budget 50\
  --init_set 200\
  --metric accuracy\
  --device cuda\
