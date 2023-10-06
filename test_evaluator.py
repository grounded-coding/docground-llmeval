# Write a test to evaluate the evaluator

from lleval.data_collector import DummyDataCollector
from lleval.utils.utilities import convert_to_json
from lleval.evaluator import DialogEvaluator
from lleval.scorer import PromptScorer, PromptTemplate
import json

# Dataset specific input collector prepares the data in the format required by the evaluator
n_indices = 150

candidate_responses = ["Ahoi", "hello", "hi"]
sample_indices = [4,6,8]
dummy_collector = DummyDataCollector()

reference_responses, turn_historys, knowledge_contexts = dummy_collector.collect_sample_contexts(sample_indices)

data = convert_to_json(output_list=candidate_responses, src_list=turn_historys, context_list=knowledge_contexts)

prompt_template = PromptTemplate(prompt_config_file="lleval/configs/prompt_likert_config.json")
llama2local = PromptScorer(api_url="http://gpu-19.apptek.local:8080/generate", metric_config_file="lleval/configs/gen_config.json", prompt_template=prompt_template, num_retries=3)
evaluator = DialogEvaluator(llama2local, dimension_definitions_file="lleval/configs/dimension_definitions.json")
eval_scores, eval_expls = evaluator.evaluate(data, print_result=True)
# eval scores is a list of dictionaries with the following keys: appropriate, accurate, overall (in the example of the code above) and numeric values as values
# eval expls is a list of dictionaries with the following keys: appropriate, accurate, overall (in the example of the code above) and strings as values

# Assert that eval_scores is a list of numeric values corresponding to the length of candidate_responses with one value for each dimension per response
assert len(eval_scores) == len(candidate_responses)
print(eval_scores)