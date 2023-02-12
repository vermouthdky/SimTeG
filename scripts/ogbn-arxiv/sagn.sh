dataset='ogbn-arxiv'
model_type='SAGN'
suffix='main'

# training parameters
eval_interval=5
lr=1e-3
weight_decay=0.0
batch_size=5000
eval_batch_size=5000
epochs=5
accum_interval=5
hidden_dropout_prob=0.1

# model parameters
gnn_num_layers=4
gnn_type=SAGN
gnn_dropout=0.2

bash scripts/train.sh $model_type $dataset $suffix \
    $eval_interval \
    $lr \
    $weight_decay \
    $batch_size \
    $eval_batch_size \
    $epochs \
    $accum_interval \
    $hidden_dropout_prob \
    $gnn_num_layers \
    $gnn_type \
    $gnn_dropout
