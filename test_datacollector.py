from data_preparation.datacollector import DSTCDataCollector
import json
from tqdm import tqdm 
import random
from typing import List, Dict


if __name__ == "__main__":
    n_indices = 10

    candidate_responses = []
    sample_indices = []
    with open("../dstc11-track5/data/val/labels.json") as f:
        data = json.load(f)
        for i in range(n_indices):
            if data[i]["target"] == True:
                candidate_responses.append(data[i]["response"])
                sample_indices.append(i)

    dstc_collector = DSTCDataCollector(base_path="../dstc11-track5/data")

    reference_responses, turn_historys, knowledge_contexts = dstc_collector.collect_sample_contexts(sample_indices)

    print(reference_responses)
    print(turn_historys)
    print(knowledge_contexts)