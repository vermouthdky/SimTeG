dataset='ogbn-arxiv'
model_type='GBert'
lm_type='Deberta'
gnn_type='GAMLP'
suffix="adapter_"${lm_type}"_"${gnn_type}

bash scripts/train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --eval_interval 1 \
    --lr 1e-4 \
    --weight_decay 5e-5 \
    --batch_size 40 \
    --eval_batch_size 400 \
    --epochs 10 \
    --accum_interval 5 \
    --hidden_dropout_prob 0.12 \
    --header_dropout_prob 0.35 \
    --attention_dropout_prob 0.18 \
    --label_smoothing 0.22 \
    --scheduler_warmup_ratio 0.3 \
    --use_adapter
