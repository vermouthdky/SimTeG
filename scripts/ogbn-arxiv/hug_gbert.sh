model_type='GBert'
dataset='ogbn-arxiv'
lm_type='Deberta'
gnn_type='GAMLP'
suffix=${lm_type}_${gnn_type}_main_hug
# inherit and use_adapter

bash scripts/train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --eval_patience 50000 \
    --save_ckpt_per_valid \
    --num_iterations 8 \
    --lr 5e-4 \
    --gnn_lr 5e-4 \
    --weight_decay 5e-5 \
    --gnn_weight_decay 2e-6 \
    --batch_size 20 \
    --eval_batch_size 200 \
    --accum_interval 10 \
    --hidden_dropout_prob 0.16 \
    --header_dropout_prob 0.20 \
    --label_smoothing 0.28 \
    --kl_loss_weight 0.5 \
    --kl_loss_temp 2 \
    --epochs 2 \
    --warmup_ratio 0.5 \
    --use_adapter \
    --inherit \
    --use_cache \
    --use_hug_trainer