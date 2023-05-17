dataset=ogbn-arxiv
model=revgat
lm_model_types=(all-MiniLM-L6-v2 all-mpnet-base-v2 all-roberta-large-v1 e5-large)

for i in 0 1 2 3; do
    lm_model_type=${lm_model_types[i]}

    suffix=optuna_X_${lm_model_type}

    output_dir=out/${dataset}/${model}/${suffix}
    ckpt_dir=${output_dir}/ckpt

    mkdir -p ${output_dir}
    mkdir -p ${ckpt_dir}

    bert_x_dir=out/${dataset}/${lm_model_type}/optuna/best/cached_embs/x_embs.pt

    python -m src.misc.revgat.hp_search \
        --use-norm \
        --no-attn-dst \
        --mode teacher \
        --gpu $i \
        --suffix ${suffix} \
        --use_bert_x \
        --bert_x_dir $bert_x_dir \
        --ckpt_dir $ckpt_dir \
        --output_dir $output_dir \
        2>&1 | tee ${output_dir}/log.txt &

done
