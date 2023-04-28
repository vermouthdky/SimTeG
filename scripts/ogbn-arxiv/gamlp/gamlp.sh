dataset=ogbn-arxiv
model_type=GAMLP
lm_type=all-MiniLM-L6-v2
suffix=${lm_type}/main
bert_x_dir=out/${dataset}/${suffix}/cached_embs/iter_0_x_embs.pt

bash scripts/train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --lm_type $lm_type \
    --gnn_lr 3e-3 \
    --gnn_eval_interval 5 \
    --gnn_weight_decay 2e-6 \
    --gnn_batch_size 10000 \
    --gnn_eval_batch_size 10000 \
    --gnn_epochs 100 \
    --gnn_dropout 0.35 \
    --gnn_num_layers 4 \
    --lr_scheduler_type constant \
    --gnn_label_smoothing 0.2 \
    --gnn_warmup_ratio 0.3 \
    --use_bert_x \
    --bert_x_dir $bert_x_dir \
    --n_exps 10

suffix=${lm_type}/fix
bert_x_dir=out/${dataset}/${suffix}/cached_embs/iter_0_x_embs.pt

bash scripts/train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --lm_type $lm_type \
    --gnn_lr 3e-3 \
    --gnn_eval_interval 5 \
    --gnn_weight_decay 2e-6 \
    --gnn_batch_size 10000 \
    --gnn_eval_batch_size 10000 \
    --gnn_epochs 100 \
    --gnn_dropout 0.35 \
    --gnn_num_layers 4 \
    --lr_scheduler_type constant \
    --gnn_label_smoothing 0.2 \
    --gnn_warmup_ratio 0.3 \
    --use_bert_x \
    --bert_x_dir $bert_x_dir \
    --n_exps 10

lm_type=all-mpnet-base-v2
suffix=${lm_type}/main
bert_x_dir=out/${dataset}/${suffix}/cached_embs/iter_0_x_embs.pt

bash scripts/train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --lm_type $lm_type \
    --gnn_lr 3e-4 \
    --gnn_eval_interval 5 \
    --gnn_weight_decay 4e-5 \
    --gnn_batch_size 10000 \
    --gnn_eval_batch_size 10000 \
    --gnn_epochs 100 \
    --gnn_dropout 0.38 \
    --gnn_num_layers 5 \
    --lr_scheduler_type constant \
    --gnn_label_smoothing 0.5 \
    --gnn_warmup_ratio 0.1 \
    --use_bert_x \
    --bert_x_dir $bert_x_dir \
    --n_exps 10

suffix=${lm_type}/fix
bert_x_dir=out/${dataset}/${suffix}/cached_embs/iter_0_x_embs.pt

bash scripts/train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --lm_type $lm_type \
    --gnn_lr 3e-4 \
    --gnn_eval_interval 5 \
    --gnn_weight_decay 4e-5 \
    --gnn_batch_size 10000 \
    --gnn_eval_batch_size 10000 \
    --gnn_epochs 100 \
    --gnn_dropout 0.38 \
    --gnn_num_layers 5 \
    --lr_scheduler_type constant \
    --gnn_label_smoothing 0.5 \
    --gnn_warmup_ratio 0.1 \
    --use_bert_x \
    --bert_x_dir $bert_x_dir \
    --n_exps 10

lm_type=all-roberta-large-v1
suffix=${lm_type}/main
bert_x_dir=out/${dataset}/${suffix}/cached_embs/iter_0_x_embs.pt

bash scripts/train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --lm_type $lm_type \
    --gnn_lr 7e-4 \
    --gnn_eval_interval 5 \
    --gnn_weight_decay 8e-7 \
    --gnn_batch_size 10000 \
    --gnn_eval_batch_size 10000 \
    --gnn_epochs 100 \
    --gnn_dropout 0.72 \
    --gnn_num_layers 4 \
    --lr_scheduler_type constant \
    --gnn_label_smoothing 0.55 \
    --gnn_warmup_ratio 0.17 \
    --use_bert_x \
    --bert_x_dir $bert_x_dir \
    --n_exps 10

suffix=${lm_type}/fix
bert_x_dir=out/${dataset}/${suffix}/cached_embs/iter_0_x_embs.pt

bash scripts/train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --lm_type $lm_type \
    --gnn_lr 7e-4 \
    --gnn_eval_interval 5 \
    --gnn_weight_decay 8e-7 \
    --gnn_batch_size 10000 \
    --gnn_eval_batch_size 10000 \
    --gnn_epochs 100 \
    --gnn_dropout 0.72 \
    --gnn_num_layers 4 \
    --lr_scheduler_type constant \
    --gnn_label_smoothing 0.55 \
    --gnn_warmup_ratio 0.17 \
    --use_bert_x \
    --bert_x_dir $bert_x_dir \
    --n_exps 10
