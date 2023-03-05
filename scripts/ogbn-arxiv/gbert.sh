dataset='ogbn-arxiv'
model_type='GBert'
lm_type='Deberta'
gnn_type='GAMLP'
suffix=${lm_type}'_adapter_'${gnn_type}"_lr1e-4"

# training parameters
eval_interval=1
lr=1e-4 # 2e-5 on 3090
# lr=1e-4
weight_decay=5e-5
batch_size=40 # on A100; 10 on 3090
eval_batch_size=400
epochs=10
accum_interval=5
hidden_dropout_prob=0.12
header_dropout_prob=0.35
attention_dropout_prob=0.18
label_smoothing=0.22
scheduler_warmup_ratio=0.3
use_adapter=true
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

use_adapter=false
suffix=${lm_type}'_'${gnn_type}"_lr1e-4"

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
