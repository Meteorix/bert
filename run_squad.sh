# BERT_BASE_DIR=/data/nfsdata/nlp/BERT_BASE_DIR/uncased_L-12_H-768_A-12
SQUAD_DIR=/data/nfsdata/nlp/datasets/reading_comprehension/squad
# OUTPUT_DIR=/data/nfsdata/data/liuxin/squad_base

BERT_BASE_DIR=/data/nfsdata/nlp/BERT_BASE_DIR/uncased_L-12_H-768_A-12
OUTPUT_DIR=/home/liuxin/TensorRT/demo/BERT/squad_output_path_yuxian

python run_squad.py \
      --vocab_file=$BERT_BASE_DIR/vocab.txt \
      --bert_config_file=$BERT_BASE_DIR/bert_config.json \
      --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
      --do_train=False \
      --train_file=$SQUAD_DIR/train-v1.1.json \
      --do_predict=True \
      --predict_file=$SQUAD_DIR/dev-v1.1.json \
      --train_batch_size=12 \
      --predict_batch_size=32 \
      --learning_rate=3e-5 \
      --num_train_epochs=2.0 \
      --max_seq_length=384 \
      --doc_stride=128 \
      --output_dir=$OUTPUT_DIR


# python $SQUAD_DIR/evaluate-v1.1.py $SQUAD_DIR/dev-v1.1.json $OUTPUT_DIR/predictions.json
