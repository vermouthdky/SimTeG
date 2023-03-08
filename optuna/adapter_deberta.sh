model_type='Deberta'
dataset='ogbn-arxiv'
suffix='adapter'

# fixed training parameters
batch_size=20
eval_batch_size=200
eval_interval=1

# set distributed env
WORLD_SIZE=4
MASTER_PORT=32020

project_dir='.'
output_dir=${project_dir}/out/${dataset}/${model_type}/${suffix}
ckpt_dir=${output_dir}/ckpt

mkdir -p ${output_dir}
mkdir -p ${ckpt_dir}

torchrun --nproc_per_node $WORLD_SIZE --master_port $MASTER_PORT run_optuna.py \
    --model_type $model_type \
    --dataset $dataset \
    --suffix $suffix \
    --ckpt_dir $ckpt_dir \
    --batch_size $batch_size \
    --eval_batch_size $eval_batch_size \
    --eval_interval $eval_interval \
    --use_adapter \
    --output_dir $output_dir 2>&1 | tee ${output_dir}/log.txt
