#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import os
import numpy as np

class GeometricSolver:
    def __init__(self, model_path: str, emb_dim: int, k_neighbors: int = 50, k_results: int = 25, device: str = "cuda"):
        self.model_path = model_path
        self.emb_dim = emb_dim
        self.device = device
        self.k_n = k_neighbors
        self.k_r = k_results
        self._load_embeddings()

    def _load_embeddings(self) -> None:
        self.entity_embeddings = torch.from_numpy(np.load(os.path.join(self.model_path, 'entity_embedding.npy'))).to(self.device)
        self.relation_embeddings = torch.from_numpy(np.load(os.path.join(self.model_path, 'relation_embedding.npy'))).to(self.device)

    def get_metrics(self) -> dict:
        if not hasattr(self, 'metrics'):
            raise ValueError("No metrics available. Execute at least one query with true answers to get metrics.")
        
        # summary = {}
        # for step in self.metrics:
        #     summary[step] = {}
        #     for metric in self.metrics[step]:
        #         if self.metrics[step][metric]:
        #             summary[step][metric] = sum(self.metrics[step][metric]) / len(self.metrics[step][metric])
        #         else:
        #             summary[step][metric] = 0.0
        # return summary

        return self.metrics

    def set_k(self, k_neighbors: int = None, k_results: int = None) -> None:
        if k_neighbors is not None:
            self.k_n = k_neighbors
        if k_results is not None:
            self.k_r = k_results

    #TODO: predict in futuro dovrà implementare anche RotatE ecc
    def _predict(self, head_id: int, relation_id: int, tail_id: int, mode: str = "tail-batch", last: bool = False) -> tuple[torch.tensor, torch.tensor]:
        head = self.entity_embeddings[head_id]
        rel = self.relation_embeddings[relation_id]
        tail = self.entity_embeddings[tail_id]

        if mode == "head-batch":
            target = tail - rel
        else:
            target = head + rel

        # L1 distance to all entities
        distances = torch.norm(self.entity_embeddings - target, p=2, dim=1)

        # - to get largest scores
        best_ids = torch.topk(-distances, self.k_r if last else self.k_n).indices
        return best_ids, distances[best_ids]

    def _project(self, query: list, mode: str = "tail-batch", last: bool = False) -> list[list]:
        acc = []

        for q in query:
            h, r = q

            ids, _ = self._predict(int(h), int(r), int(h), mode=mode, last=last)
            acc.append(ids.cpu().tolist())

        # return acc if len(acc) > 1 else acc[0] # TODO: gestire se è singolo elemento in execute_query
        return acc

    def _union(self, list_of_lists: list[list]) -> set:
        return set(np.array(list_of_lists).flatten())
    
    def _intersection(self, list_of_lists: list[list]) -> set:
        if not list_of_lists:
            return set()
        
        result = set(list_of_lists[0])
        for lst in list_of_lists[1:]:
            result &= set(lst)
        return result
    
    def _initialize_metrics(self) -> None:
        self.metrics = {
            "first": { 
                 "mrr": [],
                "hits3": [],
                "hits5": [],
                "hits10": [],
            },
            "second": {
                "mrr": [],
                "hits3": [],
                "hits5": [],
                "hits10": [],
            }
        }

    def _evaluate_query(self, predicted: set, true: set, step: str = "first") -> None:
        ids = torch.tensor(list(predicted))

        for t in true:
            ranking = (ids == t)
            if ranking.sum():
                ranking = ranking.nonzero(as_tuple=True)[0]+1
                self.metrics[step]['mrr'].append(1.0 / ranking.item())
                self.metrics[step]['hits3'].append(1.0 if ranking <= 3 else 0.0)
                self.metrics[step]['hits5'].append(1.0 if ranking <= 5 else 0.0)
                self.metrics[step]['hits10'].append(1.0 if ranking <= 10 else 0.0)

    def execute_query(self, query: list, proj_mode: str = "inter", agg_mode: str = "union", trues: list[set] = None):
        if trues:
            self._initialize_metrics()

        proj_agg = self._union if proj_mode == "union" else self._intersection
        res_agg = self._union if agg_mode == "union" else self._intersection

        final_ids = set()
        intermediate_ids = []

        if len(query) > 1 and isinstance(query[-1], np.int64):
            rel_target = query[-1]

        if len(query) == 1:
            projections = self._project(query, mode="tail-batch")
            final_ids = proj_agg(projections)
            if trues:
                self._evaluate_query(final_ids, trues, step="first")

            return final_ids

        projections = self._project(query[0], mode="tail-batch")
        intermediate_ids = proj_agg(projections)
        print(f"Intermediate ids ({len(intermediate_ids)})")
        
        if trues:
            self._evaluate_query(intermediate_ids, trues[0], step="first")

        final_queries = [ (node, rel_target) for node in intermediate_ids ]

        projections = self._project(final_queries, mode="tail-batch", last=True)
        final_ids = res_agg(projections)

        if trues:
            print(len(final_ids), len(trues[1]))
            self._evaluate_query(final_ids, trues[1], step="second")
        
        # TODO: Manca gestire per query più complesse (adesso solo  1p, np + inter/union, np + inter/union + 1p + inter/union)
        # TODO: Implementare anche l'evaluation (optional) usando i trues ids. Fare gestione esterna richiamata in questa funzione

        return final_ids