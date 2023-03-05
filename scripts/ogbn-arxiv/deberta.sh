dataset='ogbn-arxiv'
model_type='Deberta'
suffix='main'
lm_type='Deberta'
gnn_type='GAMLP'

# training parameters
eval_interval=1
lr=2e-5
weight_decay=5e-5
batch_size=20
eval_batch_size=100
epochs=10
accum_interval=5
hidden_dropout_prob=0.12
header_dropout_prob=0.35
attention_dropout_prob=0.18
label_smoothing=0.22
scheduler_warmup_ratio=0.3
use_adapter=false
use_SLE=false
use_bert_x=false

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
    $use_bert_x \
    $lm_type \
    $gnn_type

dataset='ogbn-arxiv'
model_type='Deberta'
suffix='reproduct_GLEM'
lm_type='Deberta'
gnn_type='GAMLP'

# training parameters
eval_interval=1
lr=2e-5
weight_decay=5e-5
batch_size=36
eval_batch_size=360
epochs=10
accum_interval=5
hidden_dropout_prob=0.3
header_dropout_prob=0.4
attention_dropout_prob=0.1
label_smoothing=0.3
scheduler_warmup_ratio=0.3
use_adapter=false
use_SLE=false
use_bert_x=false

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
    $use_bert_x \
    $lm_type \
    $gnn_type
