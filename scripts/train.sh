# mode, model, dataset
model_type=$2
dataset=$4
suffix=$6

# distributed training envs
WORLD_SIZE=4
MASTER_PORT=32020

# set up output directory
project_dir='.'
output_dir=${project_dir}/out/${dataset}/${model_type}/${suffix}
ckpt_dir=${output_dir}/ckpt

mkdir -p ${output_dir}
mkdir -p ${ckpt_dir}

torchrun --nproc_per_node $WORLD_SIZE --master_port $MASTER_PORT main.py \
    --mode train --output_dir $output_dir --ckpt_dir $ckpt_dir \
    $@ 2>&1 | tee ${output_dir}/log.txt

# deepspeed main.py \
#     --mode train --output_dir $output_dir --ckpt_dir $ckpt_dir \
#     $@ 2>&1 | tee ${output_dir}/log.txt

# accelerate launch main.py \
#     --mode train --output_dir $output_dir --ckpt_dir $ckpt_dir \
#     $@ 2>&1 | tee ${output_dir}/log.txt
