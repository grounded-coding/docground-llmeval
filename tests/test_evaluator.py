# Write a test to evaluate the evaluator

from utils import get_context_data, convert_to_json
from metric.evaluator import DialogEvaluator
from utils import generate_samples, evaluate_with_scorer
from metric.scorer import PromptScorer
import json

# Dataset specific input collector prepares the data in the format required by the evaluator
# json_files = [f'{dataset}/{dataset_split}/labels.json', pred_path]
# datasets = load_jsons(json_files)
# conv = get_prompt(i, split=dataset_split, label_print=False, max_turns=5, max_n_sent=0, dataset=dataset)[0]

# Candidate responses is a list of strings, turn historys is a list of lists of strings, knowledge_contexts is a list of lists of relevant document strings
candidate_responses = ["Yes I can. The Cambridge Guest House is a great place to stay. It has free wifi and is in the north of town.",
                       ""]
# Example data
turn_historys = [["Hi, I'm looking for a guest house in the north of town. Is there one with free wifi?"]]
knowledge_contexts = [[":R: (Cambridge Guest House) Great guest house north of the town.", ":F: (Cambridge Guest House) Do you have free wifi? Yes, we have free wifi."]]

data = convert_to_json(output_list=candidate_responses, src_list=turn_historys, context_list=knowledge_contexts)

llama2local = PromptScorer(api_url="http://gpu-19.apptek.local:8080/generate", metric_config_file="metric_config.json", num_retries=3)
evaluator = DialogEvaluator(llama2local)
eval_scores = evaluator.evaluate(data, print_result=True)

# Assert that eval_scores is a list of numeric values corresponding to the length of candidate_responses with one value for each dimension per response
assert len(eval_scores) == len(candidate_responses)
print(eval_scores)

# save_files(pred_acc_count, pred_app_count, len(samples), pred_path, winexpls, method)