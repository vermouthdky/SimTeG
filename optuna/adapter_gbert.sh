model_type='GBert'
dataset='ogbn-arxiv'
lm_type='Deberta'
gnn_type='GAMLP'
suffix=optuna_${lm_type}_${gnn_type}_adapter_inherit_kl

bash optuna/run.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --eval_interval 1 \
    --save_ckpt_per_valid \
    --num_iterations 8 \
    --lr 5e-4 \
    --weight_decay 5e-5 \
    --batch_size 20 \
    --eval_batch_size 200 \
    --accum_interval 5 \
    --hidden_dropout_prob 0.16 \
    --label_smoothing 0.28 \
    --use_adapter \
    --inherit
