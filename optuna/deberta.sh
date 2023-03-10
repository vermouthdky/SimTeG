model_type='Deberta'
dataset='ogbn-arxiv'
suffix='main'

bash optuna/run.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --batch_size 20 \
    --eval_batch_size 200 \
    --eval_interval 1
