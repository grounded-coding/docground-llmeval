# Write a test to evaluate the evaluator

from utils.utilities import convert_to_json
from metric.evaluator import DialogEvaluator
from metric.scorer import PromptScorer, PromptTemplate
import json

# Dataset specific input collector prepares the data in the format required by the evaluator

# Candidate responses is a list of strings, turn historys is a list of lists of strings, knowledge_contexts is a list of lists of relevant document strings
candidate_responses = ["Yes I can. The Cambridge Guest House is a great place to stay. It has free wifi and is in the north of town.",
                       "Sorry I can't. Also I have no idea what you are talking about you stupid human."]
# Example data
turn_historys = [["Hi, I'm looking for a guest house in the north of town. Is there one with free wifi?"],
                 ["Hello sir can you halp me?"]]
knowledge_contexts = [[":R: (Cambridge Guest House) Great guest house north of the town.", ":F: (Cambridge Guest House) Do you have free wifi? Yes, we have free wifi."],
                      []]

data = convert_to_json(output_list=candidate_responses, src_list=turn_historys, context_list=knowledge_contexts)

prompt_template = PromptTemplate()
llama2local = PromptScorer(api_url="http://gpu-19.apptek.local:8080/generate", metric_config_file="metric_likert_config.json", prompt_template=prompt_template, num_retries=3)
evaluator = DialogEvaluator(llama2local)
eval_scores = evaluator.evaluate(data, print_result=True, expls=True)

# Assert that eval_scores is a list of numeric values corresponding to the length of candidate_responses with one value for each dimension per response
assert len(eval_scores) == len(candidate_responses)
print(eval_scores)