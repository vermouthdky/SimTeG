dataset=$1
lm_model_type=$2
gnn_model_type=$3
bert_x_dir=$4

suffix=optuna_on_X_${lm_model_type}
bash scripts/optuna.sh --model_type $gnn_model_type --dataset $dataset --suffix $suffix \
    --lm_type $lm_model_type \
    --gnn_eval_interval 5 \
    --gnn_batch_size 10000 \
    --gnn_eval_batch_size 10000 \
    --gnn_epochs 100 \
    --lr_scheduler_type constant \
    --use_bert_x \
    --bert_x_dir $bert_x_dir \
    --n_trials 20
