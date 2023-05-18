# bash scripts/hp_search/lm.sh ogbn-products all-MiniLM-L6-v2
# bash scripts/hp_search/lm.sh ogbn-products all-mpnet-base-v2
# bash scripts/hp_search/lm.sh ogbn-products all-roberta-large-v1
bash scripts/hp_search/lm.sh ogbn-products e5-large

for lm in all-MiniLM-L6-v2 all-mpnet-base-v2 all-roberta-large-v1 e5-large; do
    bash scripts/hp_search/gnn.sh ogbn-products $lm GAMLP &
    bash scripts/hp_search/gnn.sh ogbn-products $lm SAGN &
    bash scripts/hp_search/gnn.sh ogbn-products $lm SGC &
    wait
    echo "Finished GAMLP SAGN SGC on X-$lm"
done
