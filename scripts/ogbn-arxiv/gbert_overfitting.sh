model_type='GBert'
dataset='ogbn-arxiv'
lm_type='Deberta'
gnn_type='GAMLP'

for hidden_dropout in 0.2 0.3 0.4 0.5; do
    for hidden_size in 16 32 64; do
        suffix=${lm_type}_${gnn_type}_hidden_dropout_${hidden_dropout}_hidden_size_${hidden_size}
        bash scripts/train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
            --eval_interval 1 \
            --save_ckpt_per_valid \
            --num_iterations 8 \
            --lr 5e-4 \
            --gnn_lr 1e-2 \
            --weight_decay 1e-4 \
            --gnn_weight_decay 2e-6 \
            --batch_size 20 \
            --eval_batch_size 200 \
            --accum_interval 1 \
            --hidden_dropout_prob $hidden_dropout \
            --header_dropout_prob 0.12 \
            --label_smoothing 0.32 \
            --adapter_hidden_size $hidden_size \
            --kl_loss_weight 2.0 \
            --kl_loss_temp 1 \
            --epochs 2 \
            --warmup_ratio 0.15 \
            --use_hug_trainer \
            --gnn_lr 5e-3 \
            --gnn_eval_interval 5 \
            --gnn_weight_decay 1e-7 \
            --gnn_batch_size 10000 \
            --gnn_eval_batch_size 10000 \
            --gnn_epochs 100 \
            --gnn_dropout 0.15 \
            --gnn_label_smoothing 0.5 \
            --use_adapter \
            --fix_gnn \
            --gnn_inherit \
            --inherit \
            --compute_kl_loss \
            --lr_scheduler_type constant
    done
done
