dataset='ogbn-arxiv'
model_type='Deberta'
suffix='main'

bash scripts/train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --eval_interval 1 \
    --lr 2e-5 \
    --weight_decay 5e-5 \
    --batch_size 20 \
    --eval_batch_size 200 \
    --epochs 10 \
    --accum_interval 1 \
    --hidden_dropout_prob 0.12 \
    --header_dropout_prob 0.35 \
    --attention_dropout_prob 0.18 \
    --label_smoothing 0.22 \
    --schedular_warmup_ratio 0.3
