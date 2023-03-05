dataset='ogbn-arxiv'
model_type='GAMLP'
suffix='bert_x_JK_GAMLP'

# training parameters
# searched by optuna, valid acc: 70.30 %
eval_interval=5
lr=0.01
weight_decay=1e-7
batch_size=10000
eval_batch_size=10000
epochs=500
accum_interval=1
hidden_dropout_prob=0.12
header_dropout_prob=0.15
attention_dropout_prob=0.18
label_smoothing=0.5
scheduler_warmup_ratio=0.3
use_adapter=false
use_SLE=false
use_bert_x=true

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
    $scheduler_warmup_ratio \
    $use_adapter \
    $use_SLE \
    $use_bert_x
