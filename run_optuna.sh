model_type='SAGN'
dataset='ogbn-arxiv'
suffix='optuna'

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
    --ckpt_dir $ckpt_dir \
    --weight_decay 0.0 \
    --epochs 10 \
    --batch_size 1000 \
    --eval_batch_size 10000 \
    --eval_interval 5 \
    --gnn_num_layers 4 \
    --gnn_type SAGN \
    --gnn_dropout 0.2 \
    --output_dir $output_dir | tee ${output_dir}/log.txt
