# distributed training envs
WORLD_SIZE=6
MASTER_PORT=32020

# training parameters
ckpt_dir='./ckpt'
eval_interval=5
lr=1e-4
weight_decay=0.0
batch_size=5
epochs=50
accum_interval=5

# model parameters
gnn_num_layers=2
gnn_type=GraphSAGE
gnn_dropout=0.2

python -m torch.distributed.run --nproc_per_node $WORLD_SIZE --master_port $MASTER_PORT main.py \
    --ckpt_dir $ckpt_dir \
    --eval_interval $eval_interval \
    --lr $lr \
    --weight_decay $weight_decay \
    --batch_size $batch_size \
    --epochs $epochs \
    --accum_interval $accum_interval \
    --gnn_num_layers $gnn_num_layers \
    --gnn_type $gnn_type \
    --gnn_dropout $gnn_dropout
