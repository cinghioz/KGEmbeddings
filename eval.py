import sys
sys.path.insert(0, '/home/cc/phd/KGEmbeddings/codes')

import numpy as np
import torch
import os
from collections import defaultdict
import random
from tqdm import tqdm
import pickle 

from codes.model import KGEModel
from codes.dataloader import TrainDataset, TestDataset
from codes.triplets import TripletsEngine

EMBEDDING_DIM = 512
DATA = "primekg"
MODEL_PATH = f"/home/cc/phd/KGEmbeddings/models/TransE_{DATA}_0"
DICTS_DIR = f'/home/cc/phd/KGEmbeddings/data/{DATA}'

# chat con interessante discorso su questo: https://chatgpt.com/c/68c0325c-18c8-832b-b7fe-3eb459d9c9b8
# TODO: Implementare predict cont RotatE

def predict(head_id, relation_id, tail_id, entity_embeddings, relation_embeddings, mode = "tail-batch", top_k=10):
    head = entity_embeddings[head_id]
    rel = relation_embeddings[relation_id]
    tail = entity_embeddings[tail_id]

    if mode == "head-batch":
        target = tail - rel
    else:
        target = head + rel

    # L distance to all entities
    distances = torch.norm(entity_embeddings - target, p=2, dim=1)

    # - to get largest scores
    best_ids = torch.topk(-distances, top_k).indices
    return best_ids, distances[best_ids]

def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

def intersection(list_of_lists):
    if not list_of_lists:
        return set()
    result = set(list_of_lists[0])
    for lst in list_of_lists[1:]:
        result &= set(lst)
    return result

if __name__== "__main__":
    random.seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    entity_embedding = torch.from_numpy(np.load(os.path.join(MODEL_PATH, 'entity_embedding.npy')))
    relation_embedding = torch.from_numpy(np.load(os.path.join(MODEL_PATH, 'relation_embedding.npy')))

    number_of_entities = entity_embedding.shape[0]
    number_of_relations = relation_embedding.shape[0]

    args = {
        "model": "TransE",
        "hidden_dim": EMBEDDING_DIM,
        "gamma": 24.0,
        "double_entity_embedding": False,
        "double_relation_embedding": False,
        "do_train": False,
        "test_batch_size": 512,
        "cpu_num": 32,
        "cuda": True,
        "test_log_steps": 1000,
        "nentity": number_of_entities,
        "nrelation": number_of_relations,
        "mode": "tail-batch",
        "device": device
    }

    class DictToObject:
        def __init__(self, dictionary):
            for key, value in dictionary.items():
                setattr(self, key, value)

    args = DictToObject(args)

    kge_model = KGEModel(
        model_name=args.model,
        nentity=number_of_entities,
        nrelation=number_of_relations,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding
    ).to(device)

    print("Loading checkpoint...")
    checkpoint = torch.load(os.path.join(MODEL_PATH, 'checkpoint'))
    init_step = checkpoint['step']
    kge_model.load_state_dict(checkpoint['model_state_dict'])

    if args.do_train:
        current_learning_rate = checkpoint['current_learning_rate']
        warm_up_steps = checkpoint['warm_up_steps']
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    kg = TripletsEngine(os.path.join(DICTS_DIR), from_splits=True, ext="csv")

    n = 100000
    ids = np.random.randint(0, len(kg.triplets), size=n)
    mrr = []
    recall = []

    metrics = {
        'MRR': [],
        'HITS@1': [],
        'HITS@3': [],
        'HITS@10': [],
        'HITS@25': [],
    }

    kge_model.eval()

    if args.mode == 'head-batch':
        adj = {k: torch.tensor(v, device=device) for k, v in kg.t2h.items()}
    else:
        adj = {k: torch.tensor(v, device=device) for k, v in kg.h2t.items()}

    for id in tqdm(ids):
        target_head, target_relation, target_tail = kg.triplets[id]

        if args.mode == 'head-batch':
            targets = kg.t2h[(target_tail, target_relation)]
        else:
            targets = kg.h2t[(target_head, target_relation)]

        try:
            res = kge_model.single_test_step(kge_model, adj, (target_head, target_relation, target_tail), args)
            metrics['MRR'].append(res['MRR'])
            metrics['HITS@1'].append(res['HITS@1'])
            metrics['HITS@3'].append(res['HITS@3'])
            metrics['HITS@10'].append(res['HITS@10'])
            metrics['HITS@25'].append(res['HITS@25'])

        except AssertionError as error:
            print("WARNING: triple ", (target_head, target_relation, target_tail))

        top_ids, dists = predict(int(target_head), int(target_relation), int(target_tail), entity_embedding, relation_embedding, mode=args.mode, top_k=max(15, int(len(targets)*1.5)))

        recall.append(torch.isin(top_ids, torch.tensor(targets)).sum().item() / len(targets))
        # print(torch.isin(top_ids, torch.tensor(targets)).sum().item() / len(targets))

print(f"Average MRR over {n} random triplets: {np.mean(metrics['MRR'])}")
print(f"Average HITS@1, HITS@3, HITS@10, HITS@25 over {n} random triplets: {np.mean(metrics['HITS@1'])}, {np.mean(metrics['HITS@3'])}, {np.mean(metrics['HITS@10'])}, {np.mean(metrics['HITS@25'])}")
print(f"Average Recall over {n} random triplets: {np.mean(recall)}")