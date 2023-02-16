dataset='ogbn-arxiv'
model_type='Deberta'
suffix='main'

# training parameters
eval_interval=1
lr=1e-5
weight_decay=0.0
batch_size=10
eval_batch_size=100
epochs=10
accum_interval=1
hidden_dropout_prob=0.3
header_dropout_prob=0.4
attention_dropout_prob=0.1
label_smoothing=0.3
use_adapter=false

bash scripts/train.sh $model_type $dataset $suffix \
    $eval_interval \
    $lr \
    $weight_decay \
    $batch_size \
    $eval_batch_size \
    $epochs \
    $accum_interval \
    $hidden_dropout_prob \
    $header_dropout_prob \
    $attention_dropout_prob \
    $label_smoothing \
    $use_adapter
