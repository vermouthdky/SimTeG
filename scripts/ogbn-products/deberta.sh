dataset='ogbn-products'
model_type='Deberta'
suffix=main

bash scripts/train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --lr 8e-4 \
    --weight_decay 1e-6 \
    --batch_size 20 \
    --eval_batch_size 200 \
    --accum_interval 10 \
    --hidden_dropout_prob 0.18 \
    --header_dropout_prob 0.12 \
    --label_smoothing 0.15 \
    --adapter_hidden_size 32 \
    --epochs 10 \
    --warmup_ratio 0.25 \
    --use_adapter \
    --lr_scheduler_type constant
