model_type='GBert'
dataset='ogbn-arxiv'
lm_type='Deberta'
gnn_type='GAMLP'

for kl_loss_temp in 0 1 2 4; do
    for kl_loss_weight in 0.1 0.5 1.0 2.0; do
        suffix=${lm_type}_${gnn_type}_kl_loss_temp_${kl_loss_temp}_kl_loss_weight_${kl_loss_weight}
        bash scripts/train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
            --eval_interval 1 \
            --save_ckpt_per_valid \
            --num_iterations 8 \
            --lr 8e-4 \
            --gnn_lr 1e-2 \
            --weight_decay 1e-4 \
            --gnn_weight_decay 2e-6 \
            --batch_size 20 \
            --eval_batch_size 200 \
            --accum_interval 1 \
            --hidden_dropout_prob 0.15 \
            --header_dropout_prob 0.12 \
            --label_smoothing 0.32 \
            --adapter_hidden_size 64 \
            --kl_loss_weight $kl_loss_weight \
            --kl_loss_temp $kl_loss_temp \
            --epochs 2 \
            --warmup_ratio 0.15 \
            --use_hug_trainer \
            --gnn_lr 0.01 \
            --gnn_eval_interval 5 \
            --gnn_weight_decay 1e-7 \
            --gnn_batch_size 10000 \
            --gnn_eval_batch_size 10000 \
            --gnn_epochs 20 \
            --gnn_dropout 0.15 \
            --gnn_label_smoothing 0.5 \
            --use_adapter \
            --inherit \
            --compute_kl_loss
    done
done
