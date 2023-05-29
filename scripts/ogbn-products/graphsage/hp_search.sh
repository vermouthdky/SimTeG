dataset=ogbn-products
model_type=GraphSAGE
lm_model_types=(all-MiniLM-L6-v2 all-roberta-large-v1 e5-large)

for i in 0 1 2; do
    lm_model_type=${lm_model_types[i]}
    suffix=X_${lm_model_type}
    bert_x_dir=out/${dataset}/${lm_model_type}/optuna/best/cached_embs/x_embs.pt

    bash scripts/single_gpu_optuna.sh --model_type $model_type --dataset $dataset --suffix $suffix \
        --single_gpu $i \
        --lm_type $lm_model_type \
        --gnn_batch_size 1000 \
        --gnn_eval_batch_size 5000 \
        --gnn_epochs 100 \
        --use_bert_x \
        --bert_x_dir $bert_x_dir \
        --n_trial 20 &
done
