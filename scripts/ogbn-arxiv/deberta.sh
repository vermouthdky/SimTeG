dataset='ogbn-arxiv'
model_type='Deberta'
suffix='main'

bash scripts/train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --eval_interval 1 \
    --lr 4e-5 \
    --weight_decay 4e-6 \
    --batch_size 20 \
    --eval_batch_size 200 \
    --epochs 10 \
    --accum_interval 1 \
    --hidden_dropout_prob 0.19 \
    --header_dropout_prob 0.16 \
    --attention_dropout_prob 0.18 \
    --label_smoothing 0.2 \
    --scheduler_warmup_ratio 0.1 \
    --save_ckpt_per_valid

# suffix='reproduce_GLEM'

# bash scripts/train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
#     --eval_interval 1 \
#     --lr 1e-5 \
#     --weight_decay 4e-6 \
#     --batch_size 20 \
#     --eval_batch_size 200 \
#     --epochs 5 \
#     --accum_interval 1 \
#     --hidden_dropout_prob 0.3 \
#     --header_dropout_prob 0.4 \
#     --attention_dropout_prob 0.1 \
#     --label_smoothing 0.3 \
#     --scheduler_warmup_ratio 0.1
