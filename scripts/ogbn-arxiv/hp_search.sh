# bash scripts/hp_search/lm.sh ogbn-arxiv all-MiniLM-L6-v2
# bash scripts/hp_search/lm.sh ogbn-arxiv all-mpnet-base-v2
# bash scripts/hp_search/lm.sh ogbn-arxiv all-roberta-large-v1
# bash scripts/hp_search/lm.sh ogbn-arxiv e5-large

for lm in all-MiniLM-L6-v2 all-mpnet-base-v2 all-roberta-large-v1 e5-large; do
    bash scripts/hp_search/gnn.sh ogbn-arxiv $lm GAMLP &
    bash scripts/hp_search/gnn.sh ogbn-arxiv $lm SAGN &
    wait
    echo "Finished GAMLP SAGN SGC on X-$lm"
done

# bash scripts/hp_search/lm.sh ogbn-arxiv all-roberta-large-v1

# bash scripts/hp_search/gnn.sh ogbn-arxiv all-roberta-large-v1 GAMLP &
# bash scripts/hp_search/gnn.sh ogbn-arxiv all-roberta-large-v1 SAGN &
# bash scripts/hp_search/gnn.sh ogbn-arxiv all-roberta-large-v1 SGC &
# wait
# echo "Finished GAMLP SAGN SGC on X-$lm"
