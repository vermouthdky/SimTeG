dataset='ogbn-arxiv'
model_type='Roberta'
suffix='test'

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

bash scripts/train.sh $model_type $dataset $suffix \
    $eval_interval \
    $lr \
    $weight_decay \
    $batch_size \
    $eval_batch_size \
    $epochs \
    $accum_interval \
    $gnn_num_layers \
    $gnn_type \
    $gnn_dropout
