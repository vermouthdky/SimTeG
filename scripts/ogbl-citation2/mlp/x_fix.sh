dataset=ogbl-citation2
model_type=MLP

lm_model_type=all-MiniLM-L6-v2
suffix=main_X_${lm_model_type}_fix
bert_x_dir=out/${dataset}/${lm_model_type}/fix/cached_embs/x_embs.pt
bash scripts/single_gpu_train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --n_exps 10 \
    --single_gpu 0 \
    --lm_type $lm_model_type \
    --gnn_batch_size 1000 \
    --gnn_eval_batch_size 10000 \
    --gnn_epochs 100 \
    --gnn_dropout 0.0 \
    --gnn_lr 0.0005 \
    --gnn_num_layers 3 \
    --gnn_weight_decay 2e-6 \
    --use_bert_x \
    --bert_x_dir $bert_x_dir
