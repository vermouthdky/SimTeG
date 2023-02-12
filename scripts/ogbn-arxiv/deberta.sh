dataset='ogbn-arxiv'
model_type='Deberta'
suffix='main'

# training parameters
eval_interval=1
lr=1e-4
weight_decay=0.0
batch_size=10
eval_batch_size=100
epochs=50
accum_interval=5
hidden_dropout_prob=0.1

# model parameters
gnn_num_layers=4
gnn_type=SAGN
gnn_dropout=0.1

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
