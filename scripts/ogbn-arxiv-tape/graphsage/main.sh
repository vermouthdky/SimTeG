model_type=GraphSAGE

dataset=ogbn-arxiv
lm_model_type=e5-large
suffix=ensemble_X_${lm_model_type}
bert_x_dir=out/${dataset}/${lm_model_type}/main/cached_embs/x_embs.pt
bash scripts/single_gpu_train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --n_exps 10 \
    --single_gpu 0 \
    --lm_type $lm_model_type \
    --gnn_batch_size 10000 \
    --gnn_eval_batch_size 10000 \
    --gnn_epochs 50 \
    --gnn_dropout 0.4 \
    --gnn_label_smoothing 0.4 \
    --gnn_lr 0.01 \
    --gnn_num_layers 2 \
    --gnn_weight_decay 4e-6 \
    --gnn_eval_interval 1 \
    --use_bert_x \
    --bert_x_dir $bert_x_dir &

dataset=ogbn-arxiv-tape
lm_model_type=e5-large
suffix=ensemble_X_${lm_model_type}
bert_x_dir=out/${dataset}/${lm_model_type}/main/cached_embs/x_embs.pt
bash scripts/single_gpu_train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --n_exps 10 \
    --single_gpu 1 \
    --lm_type $lm_model_type \
    --gnn_batch_size 10000 \
    --gnn_eval_batch_size 10000 \
    --gnn_epochs 50 \
    --gnn_dropout 0.4 \
    --gnn_label_smoothing 0.4 \
    --gnn_lr 0.01 \
    --gnn_num_layers 2 \
    --gnn_weight_decay 4e-6 \
    --gnn_eval_interval 1 \
    --use_bert_x \
    --bert_x_dir $bert_x_dir &

dataset=ogbn-arxiv
lm_model_type=all-roberta-large-v1
suffix=ensemble_X_${lm_model_type}
bert_x_dir=out/${dataset}/${lm_model_type}/main/cached_embs/x_embs.pt
bash scripts/single_gpu_train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --n_exps 10 \
    --single_gpu 2 \
    --lm_type $lm_model_type \
    --gnn_batch_size 10000 \
    --gnn_eval_batch_size 10000 \
    --gnn_epochs 50 \
    --gnn_dropout 0.4 \
    --gnn_label_smoothing 0.4 \
    --gnn_lr 0.01 \
    --gnn_num_layers 2 \
    --gnn_weight_decay 4e-6 \
    --gnn_eval_interval 1 \
    --use_bert_x \
    --bert_x_dir $bert_x_dir &

dataset=ogbn-arxiv-tape
lm_model_type=all-roberta-large-v1
suffix=ensemble_X_${lm_model_type}
bert_x_dir=out/${dataset}/${lm_model_type}/main/cached_embs/x_embs.pt
bash scripts/single_gpu_train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --n_exps 10 \
    --single_gpu 3 \
    --lm_type $lm_model_type \
    --gnn_batch_size 10000 \
    --gnn_eval_batch_size 10000 \
    --gnn_epochs 50 \
    --gnn_dropout 0.4 \
    --gnn_label_smoothing 0.4 \
    --gnn_lr 0.01 \
    --gnn_num_layers 2 \
    --gnn_weight_decay 4e-6 \
    --gnn_eval_interval 1 \
    --use_bert_x \
    --bert_x_dir $bert_x_dir

dataset=ogbn-arxiv
lm_model_type=e5-large
suffix=ensemble_preds
bash scripts/single_gpu_train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --n_exps 10 \
    --single_gpu 1 \
    --lm_type $lm_model_type \
    --gnn_batch_size 10000 \
    --gnn_eval_batch_size 10000 \
    --gnn_epochs 50 \
    --gnn_dropout 0.4 \
    --gnn_label_smoothing 0.4 \
    --gnn_lr 0.01 \
    --gnn_num_layers 2 \
    --gnn_weight_decay 4e-6 \
    --gnn_eval_interval 1 \
    --use_gpt_preds

# logits1=out/ogbn-arxiv/GraphSAGE/ensemble_X_e5-large/cached_embs
# logits2=out/ogbn-arxiv/GraphSAGE/ensemble_X_all-roberta-large-v1/cached_embs
# logits3=out/ogbn-arxiv/GraphSAGE/ensemble_X_e5-large-v2/cached_embs

# logits4=out/ogbn-arxiv-tape/GraphSAGE/ensemble_X_e5-large/cached_embs
# logits5=out/ogbn-arxiv-tape/GraphSAGE/ensemble_X_all-roberta-large-v1/cached_embs
# logits6=out/ogbn-arxiv-tape/GraphSAGE/ensemble_X_e5-large-v2/cached_embs

# logits7=out/ogbn-arxiv/GraphSAGE/ensemble_preds/cached_embs

# python compute_ensemble.py \
#     --list_logits "${logits1} ${logits2} ${logits4} ${logits5} ${logits7}" \
#     --weights 2 2 1 1 1 \
#     --start_seed 42
