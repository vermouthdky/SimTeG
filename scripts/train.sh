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
WORLD_SIZE=4
MASTER_PORT=32020

# training parameters
eval_interval=$4
lr=$5
weight_decay=$6
batch_size=$7
eval_batch_size=$8
epochs=$9
accum_interval=${10}
hidden_dropout_prob=${11}
header_dropout_prob=${12}
attention_dropout_prob=${13}
label_smoothing=${14}
scheduler_warmup_ratio=${15}

use_adapter=${16}
if $use_adapter; then
    use_adapter='--use_adapter'
else
    use_adapter=''
fi

use_SLE=${17}
if $use_SLE; then
    use_SLE='--use_SLE'
else
    use_SLE=''
fi

use_bert_x=${18}
if $use_bert_x; then
    use_bert_x='--use_bert_x'
else
    use_bert_x=''
fi

lm_type=${19}
gnn_type=${20}

torchrun --nproc_per_node $WORLD_SIZE --master_port $MASTER_PORT main.py \
    --mode $mode \
    --model_type $model_type \
    --dataset $dataset \
    --suffix $suffix \
    --ckpt_dir $ckpt_dir \
    --output_dir $output_dir \
    --eval_interval $eval_interval \
    --lr $lr \
    --weight_decay $weight_decay \
    --batch_size $batch_size \
    --eval_batch_size $eval_batch_size \
    --epochs $epochs \
    --accum_interval $accum_interval \
    --hidden_dropout_prob $hidden_dropout_prob \
    --header_dropout_prob $header_dropout_prob \
    --attention_dropout_prob $attention_dropout_prob \
    --label_smoothing $label_smoothing \
    --scheduler_warmup_ratio $scheduler_warmup_ratio \
    $use_adapter $use_SLE $use_bert_x \
    --lm_type $lm_type \
    --gnn_type $gnn_type \
    2>&1 | tee ${output_dir}/log.txt
