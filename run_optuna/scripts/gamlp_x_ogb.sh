model_type='GAMLP'
dataset='ogbn-arxiv'
suffix='hp_x_ogb'

bash optuna/run.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --epochs 500 \
    --batch_size 10000 \
    --eval_batch_size 10000 \
    --eval_interval 5
