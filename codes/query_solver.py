#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import os
import numpy as np

class GeometricSolver:
    def __init__(self, model_path: str, model_name: str, emb_dim: int, h2t: dict, k_neighbors: int = 50, k_results: int = 25, device: str = "cuda"):
        self.model_path = model_path
        self.model_name = model_name
        self.emb_dim = emb_dim
        self.device = device
        self.k_n = k_neighbors
        self.k_r = k_results
        self.pi = 3.14159265358979323846
        self.h2t = h2t
        self._load_embeddings()
        self._initialize_metrics()

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

    def _transe(self, head: torch.Tensor, rel: torch.Tensor, tail: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "head-batch":
            return tail - rel
        else:
            return head + rel

    def _rotate(self, head: torch.Tensor, rel: torch.Tensor, tail: torch.Tensor, mode: str) -> torch.Tensor:
        # split entity embeddings into real/imag parts
        re_head, im_head = torch.chunk(head, 2, dim=-1)
        re_tail, im_tail = torch.chunk(tail, 2, dim=-1)

        # map relation to phase in [-pi, pi]
        phase_relation = rel / (self.embedding_range.item() / self.pi) # TODO: check embedding range
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == "head-batch":
            # given tail & relation → recover head
            re_target = re_relation * re_tail + im_relation * im_tail
            im_target = re_relation * im_tail - im_relation * re_tail
        else:
            # given head & relation → predict tail
            re_target = re_head * re_relation - im_head * im_relation
            im_target = re_head * im_relation + im_head * re_relation

        # recombine real & imag into a single vector
        return torch.cat([re_target, im_target], dim=-1)

    def _calculate(self, head: torch.Tensor, rel: torch.Tensor, tail: torch.Tensor, mode: str) -> torch.Tensor:
        if self.model_name.lower() == "transe":
            return self._transe(head, rel, tail, mode)
        elif self.model_name.lower() == "rotate":
            return self._rotate(head, rel, tail, mode)
        else:
            raise NotImplementedError(f"Model {self.model_name} not implemented in GeometricSolver.")

    def _predict(self, head_id: int, relation_id: int, tail_id: int, mode: str = "tail-batch", last: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        head = self.entity_embeddings[head_id]
        rel = self.relation_embeddings[relation_id]
        tail = self.entity_embeddings[tail_id]

        target = self._calculate(head, rel, tail, mode)

        # L1 distance to all entities
        distances = torch.norm(self.entity_embeddings - target, p=2, dim=1)

        # - to get largest scores
        best_ids = torch.topk(-distances, self.k_r if last else self.k_n).indices

        return best_ids, distances[best_ids]

    # Evaluate only the final set. TODO: possibile update per valutare intermedi per query complesse composte
    # TODO: capire cosa fare se il target non viene trovato nel ranking (adesso 0.0). Perchè il numero di risultati è limitato a k_results
    # magari conviene pesare sul numero di target da trovare e il numero di risultati trovati k
    def _evaluate_query(self, predicted: set, true: set) -> None:
        for t in true:
            ranking = (predicted == t)
            if ranking.sum():
                ranking = ranking.nonzero(as_tuple=True)[0]+1
                self.metrics['mrr'].append(1.0 / ranking.item())
                self.metrics['hits3'].append(1.0 if ranking <= 3 else 0.0)
                self.metrics['hits5'].append(1.0 if ranking <= 5 else 0.0)
                self.metrics['hits10'].append(1.0 if ranking <= 10 else 0.0)
            else:
                self.metrics['mrr'].append(0.0)
                self.metrics['hits3'].append(0.0)
                self.metrics['hits5'].append(0.0)
                self.metrics['hits10'].append(0.0)

    def _project(self, query: list, mode: str = "tail-batch", last: bool = False, trues: set = None) -> list[list]:
        acc = []

        for q in query:

            h, r = q

            ids, _ = self._predict(int(h), int(r), int(h), mode=mode, last=last)

            if trues and last:
                targets = self._intersection([self.h2t.get((h, r), set()), trues])
                self._evaluate_query(ids.cpu(), targets)

            acc.append(ids.cpu().tolist())

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
            "mrr": [],
            "hits3": [],
            "hits5": [],
            "hits10": [],
        }

    def _reset_metrics(self) -> None:
        for metric in self.metrics:
            self.metrics[metric] = []

    def execute_query(self, query: list, proj_mode: str = "inter", agg_mode: str = "union", trues: list[set] = None):
        proj_agg = self._union if proj_mode == "union" else self._intersection
        res_agg = self._union if agg_mode == "union" else self._intersection

        final_ids = set()
        intermediate_ids = []

        if len(query) > 1 and isinstance(query[-1], np.int64):
            rel_target = query[-1]

        if len(query) == 1:
            projections = self._project(query, mode="tail-batch")
            final_ids = projections[0]

            return final_ids

        projections = self._project(query[0], mode="tail-batch")
        intermediate_ids = proj_agg(projections)

        # Filter only nodes that can have an out edge with rel_target (da capire se ha senso)
        final_queries = [ (node, rel_target) for node in intermediate_ids if self.h2t.get(node, rel_target) is not None ]

        # print(f"Number of inter nodes ({len(final_queries)})")

        projections = self._project(final_queries, mode="tail-batch", last=True, trues=trues[1] if trues else None)
        final_ids = res_agg(projections)
        
        # TODO: Manca gestire per query più complesse (adesso solo  1p, np + inter/union, np + inter/union + 1p + inter/union)
        # TODO: Implementare anche l'evaluation (optional) usando i trues ids. Fare gestione esterna richiamata in questa funzione

        return final_ids