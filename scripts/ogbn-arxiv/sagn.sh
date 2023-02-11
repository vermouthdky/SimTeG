# mode
mode=$1

# dataset and model type
dataset="ogbn-arxiv"
model_type="SAGN"
dataset='ogbn-arxiv'
model_type='SAGN'
suffix='bert_x'

# training parameters
eval_interval=5
lr=1e-4
weight_decay=0.0
batch_size=20
eval_batch_size=300
epochs=50
accum_interval=5

# model parameters
gnn_num_layers=2
gnn_type=GraphSAGE
gnn_dropout=0.2

# use bert_x
use_bert_x=1

# bash scripts/train.sh $model_type $dataset $suffix \
#     $eval_interval \
#     $lr \
#     $weight_decay \
#     $batch_size \
#     $eval_batch_size \
#     $epochs \
#     $accum_interval \
#     $gnn_num_layers \
#     $gnn_type \
#     $gnn_dropout \
#     $use_bert_x

# cont
cont=1
base_dir=out/${dataset}/${model_type}/${suffix}
ckpt_dir=${base_dir}/ckpt
ckpt_name=${model_type}-best.pt # TGRoberta-best.pt

python main.py \
    --mode test \
    --dataset $dataset \
    --model_type $model_type \
    --gnn_type $gnn_type \
    --gnn_num_layers $gnn_num_layers \
    --cont $cont \
    --ckpt_dir $ckpt_dir \
    --use_bert_x $use_bert_x \
    --ckpt_name $ckpt_name 2>&1 | tee ${base_dir}/test.txt
