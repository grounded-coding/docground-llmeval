from abc import ABC, abstractmethod
from typing import List, Tuple
import json
from itertools import groupby
from operator import itemgetter
import pandas as pd
import re


class DataCollector(ABC):
    def __init__(self, dataset: str, dataset_split: str, dataset_name: str):
        self.dataset = dataset
        self.dataset_split = dataset_split
        self.dataset_name = dataset_name

    def get_name(self):
        if self.dataset_name is None:
            return self.__class__.__name__
        return self.dataset_name

    @abstractmethod
    def collect_sample_contexts(self, sample_indices: List[int]) -> Tuple[
        List[int], List[List[str]], List[List[str]]]:
        """
        Collect sample contexts for the given sample indices.

        :param sample_indices: A list of response ids.
        :param dataset_split: The dataset split to use.
        :param dataset: The dataset to use.
        :return: Three lists - reference_responses, turn_historys, and knowledge_contexts.
        """
        pass


class DummyDataCollector(DataCollector):
    def __init__(self) -> None:
        super().__init__(dataset="dummy_data", dataset_split="dummy_split", dataset_name="dummy")

    def collect_sample_contexts(self, sample_indices):
        reference_responses = ["Dummy response label"] * len(sample_indices)
        turn_historys = [["Speaker A dummy question", "Speaker B dummy answer"]] * len(sample_indices)
        knowledge_contexts = [["One dummy document", "Another dummy document"]] * len(sample_indices)
        return reference_responses, turn_historys, knowledge_contexts