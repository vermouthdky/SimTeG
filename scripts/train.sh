# mode, model, dataset
mode='train'
model_type=$2
dataset=$4
suffix=$6

# set up output directory
project_dir='.'
output_dir=${project_dir}/out/${dataset}/${model_type}/${suffix}
ckpt_dir=${output_dir}/ckpt

mkdir -p ${output_dir}
mkdir -p ${ckpt_dir}

# distributed training envs
WORLD_SIZE=4
MASTER_PORT=32020

echo "torchrun --nproc_per_node $WORLD_SIZE --master_port $MASTER_PORT main.py \
    --mode $mode \
    $@ 2>&1 | tee ${output_dir}/log.txt"
