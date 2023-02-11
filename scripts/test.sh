# dataset and test mode
dataset='ogbn-arxiv'
ckpt_name='GBert-best.pt' # TGRoberta-best.pt
mode='test'

# training parameters
base_dir=out/ogbn-arxiv/GBert/adapter
ckpt_dir=${base_dir}/ckpt
model_type='GBert'

# model parameters should be the same as the training
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
    --output_dir $base_dir \
    --ckpt_name $ckpt_name 2>&1 | tee ${base_dir}/test.txt
