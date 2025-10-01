# %%
import pickle
import os
import torch
import random
import numpy as np
from tqdm import tqdm

from codes.query_solver import GeometricSolver
from codes.triplets import TripletsEngine

PATH = "/home/marco_dossena/PHD/KGEmbeddings/"
EMBEDDING_DIM = 512
DATA = "umls"
# MODEL_PATH = "/home/cc/phd/KGEmbeddings/models/TransE_FB15k_0/"
# MODEL_PATH = "/home/cc/phd/KGEmbeddings/models/RotatE_FB15k_0/"
MODEL_PATH = f"{PATH}models/TransE_{DATA}_0"
MODEL_NAME = "transe"
# DICTS_DIR = "/home/cc/phd/KGEmbeddings/data/FB15k/"
DICTS_DIR = f"{PATH}data/{DATA}"

def recall_at_k(pred, true, k):
    if len(true) == 0:
        return 1.0
    
    if k > 0:
        pred_k = pred[:max(k, len(true)+10)]
    else:
        pred_k = pred

    hits = sum([1 for p in pred_k if p in true])
    return hits / len(true)

def map_at_k(pred, true, k):
    if len(true) == 0:
        return 1.0
    
    if k > 0:
        pred_k = pred[:k]
    else:
        pred_k = pred

    hits = sum([1 for p in pred_k if p in true])
    return hits / k

if __name__ == "__main__":

    with open(f'queries/{DATA}/queries-big2.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)

    queries = loaded_dict['queries']
    results = loaded_dict['results']

    kg = TripletsEngine(os.path.join(DICTS_DIR), ext="txt" if DATA == "FB15k" else "csv", from_splits=True)
    qs = GeometricSolver(MODEL_PATH, MODEL_NAME, EMBEDDING_DIM, h2t=kg.h2t, k_neighbors=50, k_results=25, device='cuda')

    qs.set_k(k_neighbors=50, k_results=25)
    recalls = {
        "recall": [],
        "recall1": [],
        "recall5": [],
        "recall10": [],
        "recall25": [],
        "recall50": [],
    }

    maps = {
        'MAP@1': [],
        'MAP@5': [],
        'MAP@10': [],
        'MAP@25': [],
        'MAP@50': [],
    }

    cnt = 0

    for query, result in tqdm(zip(queries, results), total=len(queries)):

        res = qs.execute_query(query, proj_mode="inter", agg_mode="union", trues=result)

        if len(res) > 0:
            for k in [1, 5, 10, 25, 50]:
                recalls[f"recall{k}"].append(recall_at_k(res, result[-1], k))
                maps[f'MAP@{k}'].append(map_at_k(res, result[-1], k))

            recalls["recall"].append(recall_at_k(res, result[-1], 0))

        cnt += 1
        if cnt % 1000 == 0:
            print(f"Queries evaluated: {cnt} -> remaining queries: {len(queries) - cnt}")

            metrics = qs.get_metrics()

            print(f"Average Recall over {len(queries)} complex queries (2p1): {np.mean(recalls['recall'])}")
            print(f"Average MRR over {len(queries)} complex queries (2pi): {np.mean(metrics['mrr'])}")
            print(f"Average Recall@K over {len(queries)} complex queries (2p1): 1: {np.mean(recalls['recall1'])}, 5: {np.mean(recalls['recall5'])}, 10: {np.mean(recalls['recall10'])}, \
            25: {np.mean(recalls['recall25'])}, 50: {np.mean(recalls['recall50'])}")
            print(f"Average Hits@K over {len(queries)} complex queries (2pi): 1: {np.mean(metrics['hits1'])}, 3: {np.mean(metrics['hits3'])}, \
            5: {np.mean(metrics['hits5'])}, 10: {np.mean(metrics['hits10'])}, 25: {np.mean(metrics['hits25'])}")
            print(f"Average MAP@K over {len(queries)} complex queries (2p1): 1: {np.mean(maps['MAP@1'])}, 5: {np.mean(maps['MAP@5'])}, 10: {np.mean(maps['MAP@10'])}, \
            25: {np.mean(maps['MAP@25'])}, 50: {np.mean(maps['MAP@50'])}")

    metrics = qs.get_metrics()

    print(f"Average Recall over {len(queries)} complex queries (2p1): {np.mean(recalls['recall'])}")
    print(f"Average MRR over {len(queries)} complex queries (2pi): {np.mean(metrics['mrr'])}")
    print(f"Average Recall@K over {len(queries)} complex queries (2p1): 1: {np.mean(recalls['recall1'])}, 5: {np.mean(recalls['recall5'])}, 10: {np.mean(recalls['recall10'])}, \
    25: {np.mean(recalls['recall25'])}, 50: {np.mean(recalls['recall50'])}")
    print(f"Average Hits@K over {len(queries)} complex queries (2pi): 1: {np.mean(metrics['hits1'])}, 3: {np.mean(metrics['hits3'])}, \
    5: {np.mean(metrics['hits5'])}, 10: {np.mean(metrics['hits10'])}, 25: {np.mean(metrics['hits25'])}")
    print(f"Average MAP@K over {len(queries)} complex queries (2p1): 1: {np.mean(maps['MAP@1'])}, 5: {np.mean(maps['MAP@5'])}, 10: {np.mean(maps['MAP@10'])}, \
    25: {np.mean(maps['MAP@25'])}, 50: {np.mean(maps['MAP@50'])}")


