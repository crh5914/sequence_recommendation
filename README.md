# Alogorithms for sequence recommendation
## Algorithms
[*] PopRec: A baseline for sequence recommendation that alway recommend user with item having the most popularity(most associated interactions).
[ ] BPR-MF: Top-N recommendation with matrix factorization but optimized by BPR loss.
## Metrics
* HR@k(hit ratio)
$HR@k = \frac_{1}{N}\sum_{u\in N_u}{I(R_{u,g_u} \lt k)}$
* MRR@k(mean reciprocal rank)
$MRR = \frac_{1}{N}\sum_{u\in N_u}{\frac_{1}{R_{u,g_u}}}$
* NDCG@k
The formula is quite complex.


