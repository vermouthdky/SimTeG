dataset=ogbn-products
model_type=GAMLP
suffix=optuna
bert_x_dir=out/ogbn-products/Deberta/main/cached_embs/iter_0_x_embs.pt

bash run_optuna/scripts/run.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --gnn_eval_interval 5 \
    --gnn_batch_size 10000 \
    --gnn_eval_batch_size 10000 \
    --gnn_epochs 100 \
    --use_bert_x \
    --bert_x_dir $bert_x_dir \
    --n_trials 100
