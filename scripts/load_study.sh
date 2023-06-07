dataset=$1
model=$2

lm_model_types=(all-MiniLM-L6-v2 all-roberta-large-v1 e5-large)
for i in 0 1 2; do
    lm=${lm_model_types[i]}
    echo $lm
    suffix=X_${lm}
    # suffix=optuna_peft_on_X_${lm}

    if [ "${model}" == "revgat" ]; then
        echo revgat
        python -m src.misc.revgat.hp_search --suffix $suffix --load_study
    else
        python run_optuna.py --model_type $model --dataset $dataset --suffix $suffix \
            --lm_type $lm \
            --load_study
    fi
done
