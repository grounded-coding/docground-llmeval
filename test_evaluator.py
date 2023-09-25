# Write a test to evaluate the evaluator

from data_preparation.datacollector import DSTCDataCollector
from utils.utilities import convert_to_json
from metric.evaluator import DialogEvaluator
from metric.scorer import PromptScorer, PromptTemplate
import json

# Dataset specific input collector prepares the data in the format required by the evaluator
n_indices = 150

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

data = convert_to_json(output_list=candidate_responses, src_list=turn_historys, context_list=knowledge_contexts)

prompt_template = PromptTemplate()
llama2local = PromptScorer(api_url="http://gpu-19.apptek.local:8080/generate", metric_config_file="metric_likert_config.json", prompt_template=prompt_template, num_retries=3)
evaluator = DialogEvaluator(llama2local)
eval_scores, eval_expls = evaluator.evaluate(data, print_result=True)
# eval scores is a list of dictionaries with the following keys: appropriate, accurate, overall (in the example of the code above) and numeric values as values
# eval expls is a list of dictionaries with the following keys: appropriate, accurate, overall (in the example of the code above) and strings as values

# Assert that eval_scores is a list of numeric values corresponding to the length of candidate_responses with one value for each dimension per response
assert len(eval_scores) == len(candidate_responses)
print(eval_scores)