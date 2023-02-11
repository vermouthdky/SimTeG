# dataset and test mode
dataset='ogbn-arxiv'
model_type='Roberta'
mode='save_bert_x'
ckpt_name=${model_type}-best.pt # GBert-best.pt

# training parameters
base_dir=out/ogbn-arxiv/${model_type}/main
ckpt_dir=${base_dir}/ckpt

# !model parameters should be the same as the training
gnn_type='SAGN'
gnn_num_layers=4

# cont
cont=1

python main.py \
    --mode $mode \
    --dataset $dataset \
    --model_type $model_type \
    --gnn_type $gnn_type \
    --gnn_num_layers $gnn_num_layers \
    --cont $cont \
    --ckpt_dir $ckpt_dir \
    --ckpt_name $ckpt_name 2>&1 | tee ${base_dir}/save_bert_x.txt
