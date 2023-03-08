model_type='deberta'
dataset='ogbn-arxiv'
suffix='optuna_adapter'

bash optuna/run.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --epochs 10 \
    --batch_size 20 \
    --eval_batch_size 200 \
    --eval_interval 1 \
    --use_adapter
