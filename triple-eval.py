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
EMBEDDING_DIM = 200
DATA = "umls" 
MODEL_NAME = "RotatE"
# MODEL_PATH = "/home/cc/phd/KGEmbeddings/models/TransE_FB15k_0/"
# MODEL_PATH = "/home/cc/phd/KGEmbeddings/models/RotatE_FB15k_0/"
MODEL_PATH = f"{PATH}models/{MODEL_NAME}_{DATA}_0"
# DICTS_DIR = "/home/cc/phd/KGEmbeddings/data/FB15k/"
DICTS_DIR = f"{PATH}data/{DATA}"
MODE = "tail-batch"  # head-batch or tail-batch

kg = TripletsEngine(os.path.join(DICTS_DIR), ext="txt" if DATA.startswith("FB15k") else "csv", from_splits=True)
qs = GeometricSolver(MODEL_PATH, MODEL_NAME.lower(), EMBEDDING_DIM, h2t=kg.h2t, t2h=kg.t2h, k_neighbors=50, k_results=25, device='cuda')

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

if MODE == 'head-batch':
    adj = kg.t2h
else:
    adj = kg.h2t

cnt = 0
for triplet in tqdm(kg.triplets[kg.test_set], desc="Evaluating triplets"):
    h, r, t = triplet
    if MODE == 'head-batch':
        to_remove = set(adj.get((t, r), []))
        to_remove.discard(h)
        query = (t, r)
        true = h
    else:
        to_remove = set(adj.get((h, r), []))
        to_remove.discard(t)
        query = (h, r)
        true = t

    pred = qs.execute_search_step(query, to_remove, mode=MODE)

    for k in [1, 5, 10, 25, 50]:
        recalls[f"recall@{k}"].append(recall_at_k(pred, [true], k))
        maps[f'map@{k}'].append(map_at_k(pred, [true], k))
        hits[f'hits@{k}'].append(custom_hits_at_k(pred, [true], k))
    mrr.append(custom_mrr(pred, [true]))

    cnt += 1
    if cnt % 5000 == 1:
        print(f"Final results for {cnt} triplets:")
        print(f"Mrr: {np.mean(mrr):.4f}")
        print("-----------------------")
        print_metrics(hits, "Hits", [1, 5, 10, 25, 50])
        print_metrics(recalls, "Recall", [1, 5, 10, 25, 50])
        print_metrics(maps, "Map", [1, 5, 10, 25, 50])

print(f"Final results for {len(kg.triplets)} triplets:")
print(f"Mrr: {np.mean(mrr):.4f}")
print("-----------------------")
print_metrics(hits, "Hits", [1, 5, 10, 25, 50])
print_metrics(recalls, "Recall", [1, 5, 10, 25, 50])
print_metrics(maps, "Map", [1, 5, 10, 25, 50])

# model = torch.load(os.path.join(MODEL_PATH, 'checkpoint'), weights_only=True, map_location='cuda')


