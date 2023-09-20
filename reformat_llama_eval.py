import json
import os

def reformat_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    max_id = max([entry['id'] for entry in data])
    reformatted_data = [None] * (max_id + 1)

    for entry in data:
        if entry['app_expl'].startswith("ERROR"):
            reformatted_data[entry['id']] = None
        else:
            reformatted_data[entry['id']] = entry

    with open(file_path, 'w') as file:
        json.dump(reformatted_data, file, indent=2)

def process_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("winrate_acc_app_metrics.json"):
                file_path = os.path.join(root, file)
                reformat_json_file(file_path)

# Replace 'your_directory' with the path to the directory containing the JSON files
process_directory('/home/nhilgers/setups/dstc11-track5')