# mode, model, dataset
mode='train'
model_type=$1
dataset=$2
suffix=$3

# set up output directory
project_dir='.'
output_dir=${project_dir}/out/${dataset}/${model_type}/${suffix}
ckpt_dir=${output_dir}/ckpt

mkdir -p ${output_dir}
mkdir -p ${ckpt_dir}

# distributed training envs
WORLD_SIZE=6
MASTER_PORT=32020

# training parameters
eval_interval=$4
lr=$5
weight_decay=$6
batch_size=$7
eval_batch_size=$8
epochs=$9
accum_interval=${10}

# model parameters
gnn_num_layers=${11}
gnn_type=${12}
gnn_dropout=${13}

torchrun --nproc_per_node $WORLD_SIZE --master_port $MASTER_PORT main.py \
    --mode $mode \
    --model_type $model_type \
    --dataset $dataset \
    --ckpt_dir $ckpt_dir \
    --eval_interval $eval_interval \
    --lr $lr \
    --weight_decay $weight_decay \
    --batch_size $batch_size \
    --eval_batch_size $eval_batch_size \
    --epochs $epochs \
    --accum_interval $accum_interval \
    --gnn_num_layers $gnn_num_layers \
    --gnn_type $gnn_type \
    --gnn_dropout $gnn_dropout 2>&1 | tee ${output_dir}/log.txt
