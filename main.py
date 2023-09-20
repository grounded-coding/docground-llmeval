import json
from tqdm import tqdm 
import random
from typing import List, Dict
from scripts.data_scripts.get_prompts import get_prompt
from llama_eval.prompt_llama_eval import prompt_llama_eval
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor


def load_jsons(json_files: List[str]) -> List[Dict]:
    datasets = []
    for file in json_files:
        with open(file, "r") as f:
            datasets.append(json.load(f))
    return datasets

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

def process_sample(i, datasets, method, dataset_split, dataset):
    """
    Sample positions are randomly swapped to avoid positional bias for winrate evaluation.
    """

    pred_acc_count = 0
    pred_app_count = 0
    winexpls = []

    ref = f"\"{datasets[0][i]['response']}\""
    try:
        pred = datasets[1][i]['response']
    except:
        pred_app_count += 1
        pred_acc_count += 1
        winexpls.append({"accuracy": 1, "acc_expl": "ERROR. No response available.", "appropriateness": 1, 
                         "app_expl": "ERROR. No response available.", "id": i})
    else:
        pred = f"\"{pred[0] if isinstance(pred, list) else pred}\""
        pos_pred, response_1, response_2 = ("1", pred, ref) if random.choice([True, False]) else ("2", ref, pred)

        if method == "winrate":
            conv = get_prompt(i, split=dataset_split, label_print=False, max_turns=5, max_n_sent=0, dataset=dataset)[0]
            winner_app, expl_app = prompt_llama_eval(metric="appropriate", context=conv, response_1=response_1, response_2=response_2)
            winner_app = "p" if winner_app == pos_pred else "r"
            pred_app_count += 1 if winner_app == "p" else 0
            expl_app = expl_app.replace(f"Response {pos_pred}", "p").replace(f"Response {3 - int(pos_pred)}", "r")

            conv = get_prompt(i, split=dataset_split, label_print=False, max_turns=5, max_n_sent=12, dataset=dataset)[0]
            winner_acc, expl_acc = prompt_llama_eval(metric="accurate", context=conv, response_1=response_1, response_2=response_2)
            winner_acc = "p" if winner_acc == pos_pred else "r"
            pred_acc_count += 1 if winner_acc == "p" else 0
            expl_acc = expl_acc.replace(f"Response {pos_pred}", "p").replace(f"Response {3 - int(pos_pred)}", "r")
            
            winexpls.append({"accuracy": winner_acc, "acc_expl": expl_acc, "appropriateness": winner_app, 
                             "app_expl": expl_app, "id": i})
            
        else:
            conv = get_prompt(i, split=dataset_split, label_print=False, max_turns=5, max_n_sent=0, dataset=dataset)[0]
            winner_app, expl_app = prompt_llama_eval(metric="appropriate", method="likert", context=conv, response_1=pred)
            pred_app_count += int(winner_app)

            conv = get_prompt(i, split=dataset_split, label_print=False, max_turns=5, max_n_sent=12, dataset=dataset)[0]
            winner_acc, expl_acc = prompt_llama_eval(metric="accurate", method="likert", context=conv, response_1=pred)
            pred_acc_count += int(winner_acc)

            winexpls.append({"accuracy": winner_acc, "acc_expl": expl_acc, "appropriateness": winner_app, 
                    "app_expl": expl_app, "id": i})

    return pred_acc_count, pred_app_count, winexpls

def evaluate(pred_path, samples, method="winrate", dataset_split="val", dataset="data"):
    print(f"Evaluating {method} for prediction {pred_path}")
    json_files = [f'{dataset}/{dataset_split}/labels.json', pred_path]
    datasets = load_jsons(json_files)

    pred_acc_count = 0
    pred_app_count = 0
    winexpls = []

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(process_sample, i, datasets, method, dataset_split, dataset) for i in samples]
        for future in tqdm(as_completed(futures), total=len(samples)):
            acc, app, expls = future.result()
            pred_acc_count += acc
            pred_app_count += app
            winexpls.extend(expls)

    winexpls = sorted(winexpls, key=lambda x: x["id"])

    save_files(pred_acc_count, pred_app_count, len(samples), pred_path, winexpls, method)

if __name__ == "__main__":
    n = -1
    subsample = False

    dstc11_preds = ["pred/val/rg.t5-base-0712195429.json",
                    "pred/val/rg.t5-base-peft-0712195425.json",
                    "pred/val/rg.t5-large-peft-0712195140.json",
                    "pred/val/rg.t5-large-peft-noemb-0712195146.json"]
    dstc_11_val_samples = generate_samples(n, subsample=subsample, dataset_split="val", dataset="data")

    for file in dstc11_preds:
        evaluate(pred_path=file, samples=dstc_11_val_samples, method="likert", dataset_split="val", dataset="data")