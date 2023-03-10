dataset='ogbn-arxiv'
model_type='Deberta'
suffix='adapter'

bash scripts/train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --eval_interval 1 \
    --lr 5e-4 \
    --weight_decay 5e-5 \
    --batch_size 20 \
    --eval_batch_size 200 \
    --epochs 10 \
    --accum_interval 5 \
    --hidden_dropout_prob 0.16 \
    --header_dropout_prob 0.65 \
    --label_smoothing 0.28 \
    --scheduler_warmup_ratio 0.3 \
    --use_adapter

# suffix='accum_interval_5'

# bash scripts/train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
#     --eval_interval 1 \
#     --lr 5e-4 \
#     --weight_decay 5e-5 \
#     --batch_size 20 \
#     --eval_batch_size 200 \
#     --epochs 7 \
#     --accum_interval 5 \
#     --hidden_dropout_prob 0.16 \
#     --header_dropout_prob 0.65 \
#     --label_smoothing 0.28 \
#     --scheduler_warmup_ratio 0.5 \
#     --use_adapter
