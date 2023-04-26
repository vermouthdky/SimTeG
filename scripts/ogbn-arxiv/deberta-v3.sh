dataset='ogbn-arxiv'
model_type='DebertaV3'
suffix='main' # adapter linear_scheduler

bash scripts/train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --pretrained_repo microsoft/deberta-v3-base \
    --lr 5e-4 \
    --weight_decay 1e-4 \
    --batch_size 20 \
    --eval_batch_size 200 \
    --accum_interval 1 \
    --hidden_dropout_prob 0.20 \
    --header_dropout_prob 0.12 \
    --label_smoothing 0.32 \
    --adapter_hidden_size 32 \
    --epochs 10 \
    --warmup_ratio 0.15 \
    --use_adapter \
    --lr_scheduler_type linear

suffix='full-tune'
bash scripts/train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --pretrained_repo microsoft/deberta-v3-base \
    --lr 5e-5 \
    --weight_decay 1e-4 \
    --batch_size 20 \
    --eval_batch_size 200 \
    --accum_interval 1 \
    --hidden_dropout_prob 0.20 \
    --header_dropout_prob 0.12 \
    --label_smoothing 0.32 \
    --epochs 10 \
    --warmup_ratio 0.15 \
    --lr_scheduler_type linear

suffix='full-tune-default-config'
bash scripts/train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --pretrained_repo microsoft/deberta-v3-base \
    --lr 5e-5 \
    --weight_decay 1e-4 \
    --batch_size 20 \
    --eval_batch_size 200 \
    --accum_interval 1 \
    --hidden_dropout_prob 0.20 \
    --header_dropout_prob 0.12 \
    --label_smoothing 0.32 \
    --epochs 10 \
    --warmup_ratio 0.15 \
    --lr_scheduler_type linear \
    --use_default_config
