model_type='Deberta'
dataset='ogbn-products'
suffix=optuna_full

bash run_optuna/scripts/run.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --batch_size 20 \
    --eval_batch_size 200 \
    --eval_patience 50000 \
    --n_trials 40
