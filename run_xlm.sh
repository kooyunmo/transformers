export GLUE_DIR=./glue_data
export TASK_NAME=MRPC
            
CUDA_VISIBLE_DEVICES=0 python3 ./examples/run_glue.py \
    --model_type xlm \
    --model_name_or_path xlm-mlm-enfr-1024 \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir $GLUE_DIR/$TASK_NAME \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8   \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir ./tmp/$TASK_NAME \
    --overwrite_output_dir \
    --overwrite_cache \
