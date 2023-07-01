dataset=ogbn-arxiv
model_type=e5-large-v2
suffix=fix

bash scripts/train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --pretrained_repo sentence-transformers/${model_type} \
    --eval_batch_size 200 \
    --mode "test"

lm_type=${model_type}
bert_x_dir=out/${dataset}/${lm_type}/${suffix}/cached_embs/iter_0_x_embs.pt
