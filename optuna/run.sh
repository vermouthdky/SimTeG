model_type=$1
dataset=$2
suffix=$3

# fixed training parameters
epochs=$4
batch_size=$5
eval_batch_size=$6
eval_interval=$7

# set distributed env
WORLD_SIZE=8
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
    --epochs 10 \
    --batch_size 10 \
    --eval_batch_size 100 \
    --eval_interval 1 \
    --output_dir $output_dir 2>&1 | tee ${output_dir}/log.txt
