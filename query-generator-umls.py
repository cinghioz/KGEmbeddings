import pandas as pd
import numpy as np
import requests
from tqdm import tqdm
import os
import pickle

os.chdir("/home/cc/phd/KGEmbeddings")

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

# e_map = pd.read_json("/home/cc/phd/KGEmbeddings/data/umls/entity_map.json", typ='series').to_dict()
# r_map = pd.read_json("/home/cc/phd/KGEmbeddings/data/umls/rel_map.json", typ='series').to_dict()

# inv_e_map = {v: k for k, v in e_map.items()}
# inv_r_map = {v: k for k, v in r_map.items()}

# def get_name(dic, cui):
#     return dic.get(cui, 'N/A')

if __name__ == "__main__":

    # print("Loading MRCONSO concepts ...")
    # concepts = pd.read_csv("/home/cc/phd/KGGraphRAG/umls/MRCONSO.RRF", sep="|", header=None,  usecols=[0, 1, 14], names=["CUI", "LNG", "NAME"], low_memory=False)
    # concepts = concepts[concepts['LNG'] == 'ENG']
    # cui_to_name = dict(zip(concepts['CUI'], concepts['NAME']))

    # TARGET_RELS = ["may_treat", "contraindicated_with_disease", "manifestation_of", "causative_agent_of", "pathological_process_of",
    #             "has_finding_site", "associated_morphology_of", "clinically_associated_with", "therapeutic_class_of", "has_phenotype",
    #             "associated_with", "co-occurs_with", "has_focus", "has_component", "has_active_ingredient", "has_ingredient", "used_for",
    #             "physiologic_effect_of", "mechanism_of_action_of", "has_procedure_site", "has_direct_procedure_site", "location_of", "method_of"]

    # MAP_RELS = [r_map[r] for r in TARGET_RELS]

    # umls = pd.read_csv("/home/cc/phd/KGEmbeddings/data/umls/train.csv", low_memory=False)
    # umls_r5 = umls[umls['relation_id'].isin(MAP_RELS)]

    # Define the file paths
    train_file = 'train.csv'
    test_file = 'test.csv'
    val_file = 'valid.csv'

    # Load each CSV file into a pandas DataFrame
    df_train = pd.read_csv(PATH+'data/'+DATA+'/'+train_file)
    df_test = pd.read_csv(PATH+'data/'+DATA+'/'+test_file)
    df_val = pd.read_csv(PATH+'data/'+DATA+'/'+val_file)

    # Concatenate the DataFrames into a single one
    all_data = pd.concat([df_train, df_test, df_val], ignore_index=True)

    number_of_proj = 2
    shared_tails = all_data.groupby('tail_id')['head_id'].nunique()
    shared_tails = shared_tails[shared_tails > (number_of_proj-1)]  # tails with more than number_of_proj heads

    shared_tails = shared_tails.sample(frac=1, random_state=77)  # shufflle series

    queries = [] 
    shared_tails_inter = []
    results = []  

    if shared_tails.empty:
        print("No shared tails found with relation_id = 5")
    else:
        for shared_tail_id in tqdm(shared_tails.index, desc="Processing queries"):
            # Get the heads pointing to this shared tail
            heads = all_data[all_data['tail_id'] == shared_tail_id]['head_id'].unique()[:number_of_proj]
            if len(heads) < number_of_proj:
                continue  # need at least number_of_proj heads

            relations = all_data[all_data['tail_id'] == shared_tail_id]['relation_id'].values

            query = [[(heads[i], relations[i]) for i in range(number_of_proj)]]

            new_head_id = shared_tail_id
            new_edges = all_data[(all_data['head_id'] == new_head_id) & (all_data['relation_id'] != 0)]

            if new_edges.empty:
                continue
            
            # Group tails by relations
            relation_dict = (
                new_edges.groupby('relation_id')['tail_id']
                .apply(list)
                .to_dict()
            )
            
            for rel, tails in relation_dict.items():
                # if rel in MAP_RELS:
                queries.append(query+[rel])
                shared_tails_inter.append(shared_tail_id)
                results.append(tails)

    # string_queries = []

    # for query, sh_tail, res in tqdm(zip(queries, shared_tails_inter, results), total=len(queries), desc="Formatting and save string queries"):
    #     if get_name(cui_to_name, inv_e_map[sh_tail]) == 'N/A' or get_name(cui_to_name, inv_e_map[query[0][0][0]]) == 'N/A' or get_name(cui_to_name, inv_e_map[query[0][1][0]]) == 'N/A':
    #         continue

    #     sh_tail = get_name(cui_to_name, inv_e_map[sh_tail])
    #     h1 = get_name(cui_to_name, inv_e_map[query[0][0][0]])
    #     h2 = get_name(cui_to_name, inv_e_map[query[0][1][0]])
    #     r1 = inv_r_map[query[0][0][1]]
    #     r2 = inv_r_map[query[0][1][1]]
    #     target_rel = "is associated to" if query[1] == 5 else inv_r_map[query[1]]

    #     string_queries.append([[(h1, r1), (h2, r2)], sh_tail, target_rel, [get_name(cui_to_name, inv_e_map[r]) for r in res]])

    # queries_df = pd.DataFrame(string_queries, columns=["known_heads", "shared_tail", "target_relation", "answers"])
    # queries_df.to_csv("umls_generated_queries2.csv", index=False)

    save_dict = {
        'queries': queries,
        'results': results
    }

    with open(f'/home/cc/phd/KGEmbeddings/queries/{DATA}/queries-new.pkl', 'wb') as f:
        pickle.dump(save_dict, f)

