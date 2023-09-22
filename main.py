from utils.utilities import convert_to_json
from metric.evaluator import DialogEvaluator
import json
from tqdm import tqdm 
import random
from typing import List, Dict

def generate_samples(k: int, subsample=True, dataset_split="val", dataset="data"):
    # Set k = -1 to use all samples
    # Indices of the samples where target == True
    json_files = [f'{dataset}/{dataset_split}/labels.json']
    datasets = load_jsons(json_files)
    true_samples_indices = [i for i, entry in enumerate(datasets[0]) if entry["target"] == True] 

    # Select 'k' random samples from these
    if subsample:
        selected_sample_indices = random.sample(true_samples_indices, k)
    else:
        selected_sample_indices = true_samples_indices

    return selected_sample_indices

def save_files(pred_acc_count, pred_app_count, n, pred_path, winexpls, method):

    pred_acc = pred_acc_count / n
    pred_app = pred_app_count / n

    json_filename = pred_path.split(".json")[0] + "_" + method + '_avg_metrics.json'
    expl_filename = pred_path.split(".json")[0] + "_" + method + '_acc_app_metrics.json'

    print(f"Saving predictions to {json_filename}")

    # Save the results to a JSON file
    with open(json_filename, 'w') as f:
        json.dump({"pred_acc": pred_acc, "pred_app": pred_app}, f)

    # Save the results to a JSON file
    with open(expl_filename, 'w') as f:
        json.dump(winexpls, f)


if __name__ == "__main__":
    n = -1
    subsample = False

    candidate_responses = ["pred/val/rg.t5-base-0712195429.json"]
    dstc_11_val_samples = generate_samples(n, subsample=subsample, dataset_split="val", dataset="data")

    evaluator = DialogEvaluator()
    # Add logic to use the scorer_api for evaluation

    for resp_file in candidate_responses:
        data = get_context_data(resp_file, dataset_split="val", dataset="data")
        eval_scores = evaluator.evaluate(data, resp_file, print_result=True)

    # Evaluate and return scores for each dimension
    scores = evaluate_with_scorer(task, data, selected_dimensions, scorer_api='http://gpu-19.apptek.local:8080/generate')
    print(scores)