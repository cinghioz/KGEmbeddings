#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import os
import numpy as np
from functools import reduce

class GeometricSolver:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self._load_embeddings()

    def _load_embeddings(self) -> None:
        self.entity_embedding = torch.from_numpy(np.load(os.path.join(self.model_path, 'entity_embedding.npy'))).to(self.device)
        self.relation_embedding = torch.from_numpy(np.load(os.path.join(self.model_path, 'relation_embedding.npy'))).to(self.device)

    #TODO: predict in futuro dovrà implementare anche RotatE ecc
    def _predict(self, head_id: int, relation_id: int, tail_id: int, mode: str = "tail-batch", top_k: int = 10) -> tuple[torch.tensor, torch.tensor]:
        head = self.entity_embeddings[head_id]
        rel = self.relation_embeddings[relation_id]
        tail = self.entity_embeddings[tail_id]

        if mode == "head-batch":
            target = tail - rel
        else:
            target = head + rel

        # L1 distance to all entities
        distances = torch.norm(self.entity_embeddings - target, p=1, dim=1)

        # - to get largest scores
        best_ids = torch.topk(-distances, top_k).indices
        return best_ids, distances[best_ids]
    
    def _project(self, query: list, k_neighbors: int = 50, mode: str = "tail-batch") -> list[list]:
        acc = []

        for q in query:
            h, r = q

            ids, _ = self._predict(int(h), int(r), int(h), mode=mode, top_k=k_neighbors)
            acc.append(ids.cpu().tolist())

        return acc if len(acc) > 1 else acc[0]

    def _union(self, list_of_lists: list[list]) -> set:
        return set(np.array(list_of_lists).flatten())
    
    def _intersection(self, list_of_lists: list[list]) -> set:
        if not list_of_lists:
            return set()
        
        result = set(list_of_lists[0])
        for lst in list_of_lists[1:]:
            result &= set(lst)
        return result

    def execute_query(self, query: list[list | np.array], proj_mode: str = "inter", agg_mode: str = "union", trues: list[set] = None):
        proj_agg = self._union if proj_mode == "union" else self._intersection
        res_agg = self._union if agg_mode == "union" else self._intersection

        final_ids = set()

        if len(query) > 1 and isinstance(query[-1], np.int64):
            rel_target = query[-1]

        if len(query) == 1:
            projections = self._project(query[0], mode="tail-batch")
            final_ids = proj_agg(projections)

            return final_ids
        
        


        return final_ids