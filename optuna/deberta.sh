model_type='Deberta'
dataset='ogbn-arxiv'
suffix='optuna'

epochs=10
batch_size=3
eval_batch_size=50
eval_interval=1

bash optuna/run.sh $model_type $dataset $suffix \
    $epochs \
    $batch_size \
    $eval_batch_size \
    $eval_interval
