model_type='GAMLP'
dataset='ogbn-arxiv'
suffix='hp_x_ogb'

epochs=500
batch_size=10000
eval_batch_size=10000
eval_interval=5

bash optuna/run.sh $model_type $dataset $suffix \
    $epochs \
    $batch_size \
    $eval_batch_size \
    $eval_interval
