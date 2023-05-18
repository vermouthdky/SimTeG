model_type='GBert'
dataset='ogbn-arxiv'
lm_type='Deberta'
gnn_type='GAMLP'
suffix=optuna_${lm_type}_${gnn_type}_sle

# default search hps, some may be changes as in search space
bash run_optuna/scripts/run.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --batch_size 20 \
    --eval_interval 1 \
    --num_iterations 8 \
    --lr 8e-4 \
    --gnn_lr 1e-2 \
    --weight_decay 1e-4 \
    --gnn_weight_decay 2e-6 \
    --eval_batch_size 200 \
    --accum_interval 1 \
    --hidden_dropout_prob 0.15 \
    --header_dropout_prob 0.12 \
    --label_smoothing 0.32 \
    --adapter_hidden_size 64 \
    --kl_loss_weight 2.0 \
    --kl_loss_temp 1 \
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
    --compute_kl_loss \
    --lr_scheduler_type linear \
    --SLE_threshold 0.7 \
    --SLE_mode both \
    --use_SLE \
    --n_trials 40
