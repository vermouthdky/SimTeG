dataset='ogbn-arxiv'
model_type='GBert'
suffix='adapter'

# training parameters
eval_interval=5
lr=1e-4
weight_decay=0.0
batch_size=20
eval_batch_size=300
epochs=50
accum_interval=5
hidden_dropout_prob=0.5

# model parameters
gnn_num_layers=4
gnn_type=SAGN
gnn_dropout=0.5

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
