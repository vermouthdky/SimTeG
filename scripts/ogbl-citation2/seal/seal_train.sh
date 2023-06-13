# mode, model, dataset
model_type=$2
dataset=$4
suffix=$6

# set up output directory
project_dir='.'
output_dir=${project_dir}/out/${dataset}/${model_type}/${suffix}
ckpt_dir=${output_dir}/ckpt

mkdir -p ${output_dir}
mkdir -p ${ckpt_dir}

python -m src.misc.seal.seal_link_pred.py \
    --output_dir $output_dir --ckpt_dir $ckpt_dir \
    $@ 2>&1 | tee ${output_dir}/log.txt
