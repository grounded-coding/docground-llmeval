from data_preparation.datacollector import DSTCDataCollector
import json
from tqdm import tqdm 
import random
from typing import List, Dict


if __name__ == "__main__":
    dstc_collector = DSTCDataCollector(base_path="../dstc11-track5/data")

    sample_indices = dstc_collector.get_samples_with_target(dataset_split="val")
    reference_responses, turn_historys, knowledge_contexts = dstc_collector.collect_sample_contexts(sample_indices, dataset_split="val")

    print(reference_responses)
    print(turn_historys)
    print(knowledge_contexts)
