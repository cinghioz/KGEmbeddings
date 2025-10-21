import pickle
import os
import torch
import random
import numpy as np
from tqdm import tqdm

from codes.query_solver import GeometricSolver
from codes.triplets import TripletsEngine

# PATH = "/home/marco_dossena/PHD/KGEmbeddings/"
PATH = "/home/cc/phd/KGEmbeddings/"
EMBEDDING_DIM = 512
DATA = "umls"
MODEL_NAME = "TransE"
# MODEL_PATH = "/home/cc/phd/KGEmbeddings/models/TransE_FB15k_0/"
# MODEL_PATH = "/home/cc/phd/KGEmbeddings/models/RotatE_FB15k_0/"
MODEL_PATH = f"{PATH}models/{MODEL_NAME}_{DATA}_0"
# DICTS_DIR = "/home/cc/phd/KGEmbeddings/data/FB15k/"
DICTS_DIR = f"{PATH}data/{DATA}"

def print_metrics(metrics, name, interval):
    print(f"{name} over {interval} queries:")
    for k in interval:
        print(f"{name}@{k}:{np.mean(metrics[f'{name.lower()}@{k}']):.4f}")
    print("-----------------------")

def recall_at_k(pred, true, k):
    if len(true) == 0:
        return 1.0
    
    if k > 0:
        pred_k = pred[:max(k, len(true))]
    else:
        pred_k = pred

    hits = sum([1 for p in pred_k if p in true])
    return hits / len(true)

def map_at_k(pred, true, k):
    if len(true) == 0:
        return 1.0
    
    if k > 0:
        pred_k = pred[:max(k, len(true))]
    else:
        pred_k = pred

    hits = sum([1 for p in pred_k if p in true])
    return hits / max(k, len(true))

def custom_hits_at_k(pred, trues, k):
    if len(trues) == 0:
        return 1.0
    
    hits = []

    if len(trues) == 1:
        hits.append(1.0 if trues[0] in pred[:k] else 0.0)
    else:
        for true in trues:
            pred_set = np.setdiff1d(pred, trues[trues != true])
            hits.append(1.0 if true in pred_set[:k] else 0.0)

    return np.mean(hits)

def custom_mrr(pred, trues):
    if len(trues) == 0:
        return 1.0
    
    rr = []
    if len(trues) == 1:
        if trues[0] in pred:
            rank = np.where(pred == trues[0])[0][0] + 1
            rr.append(1.0 / rank)
        else:
            rr.append(0.0)
    else:
        for true in trues:
            pred_set = np.setdiff1d(pred, trues[trues != true])
            if true in pred_set:
                rank = np.where(pred_set == true)[0][0] + 1
                rr.append(1.0 / rank)
            else:
                rr.append(0.0)

    return np.mean(rr)

if __name__ == "__main__":

    with open(f'queries/{DATA}/queries.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)

    queries = loaded_dict['queries']
    results = loaded_dict['results']

    kg = TripletsEngine(os.path.join(DICTS_DIR), ext="txt" if DATA.startswith("FB15k") else "csv", from_splits=True)
    qs = GeometricSolver(MODEL_PATH, MODEL_NAME.lower(), EMBEDDING_DIM, h2t=kg.h2t, t2h=kg.t2h, k_neighbors=50, k_results=25, device='cuda')

    qs.set_k(k_neighbors=50, k_results=25)

    recalls = {
        "recall@1": [],
        "recall@5": [],
        "recall@10": [],
        "recall@25": [],
        "recall@50": [],
    }

    maps = {
        'map@1': [],
        'map@5': [],
        'map@10': [],
        'map@25': [],
        'map@50': [],
    }

    hits = {
        'hits@1': [],
        'hits@5': [],
        'hits@10': [],
        'hits@25': [],
        'hits@50': [],
    }

    mrr = []
    cnt = 0

    for query, result in tqdm(zip(queries, results), total=len(queries)):

        res = qs.execute_query(query, proj_mode="inter", agg_mode="union")
        cnt += 1
        # result = np.array(list(result[-1]))

        if len(res) > 0:
            for k in [1, 5, 10, 25, 50]:
                recalls[f"recall@{k}"].append(recall_at_k(res, result, k))
                maps[f'map@{k}'].append(map_at_k(res, result, k))
                hits[f'hits@{k}'].append(custom_hits_at_k(res, result, k))
            mrr.append(custom_mrr(res, result))

        if cnt % 1000 == 0:
            print(f"Final results for {cnt} complex queries:")
            print(f"Mrr: {np.mean(mrr):.4f}")
            print("-----------------------")
            print_metrics(hits, "Hits", [1, 5, 10, 25, 50])
            print_metrics(recalls, "Recall", [1, 5, 10, 25, 50])
            print_metrics(maps, "Map", [1, 5, 10, 25, 50])

    print(f"Final results for {len(queries)} complex queries:")
    print(f"Mrr: {np.mean(mrr):.4f}")
    print("-----------------------")
    print_metrics(hits, "Hits", [1, 5, 10, 25, 50])
    print_metrics(recalls, "Recall", [1, 5, 10, 25, 50])
    print_metrics(maps, "Map", [1, 5, 10, 25, 50])


