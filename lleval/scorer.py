from tqdm import tqdm

import requests
import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from lleval.evaluator import PromptTemplate

class PromptScorer:
    # This is a technical class. It receives a pre-built prompt and submits it to the local LLM API.
    def __init__(self, api_url, metric_config_file, prompt_template: PromptTemplate, num_retries=3):
        self.api_url = api_url
        self.num_retries = num_retries
        self.prompt_template = prompt_template

        with open(metric_config_file, "r") as conf:
            self.metric_config = json.load(conf)

    def build_and_submit_prompt(self, i, output_list, src_list, context_list, dimension, method="likert"):
        prompt = self.prompt_template.get_prompt(dimension, output_list[i], src_list[i], context_list[i], 
                                                 dim_description=dimension['description'])

        # print(prompt)

        data = {
            "inputs": prompt,
            "parameters": self.metric_config["gen_params"]
        }

        headers = {
            'Content-Type': 'application/json'
        }

        success = False
        for _ in range(self.num_retries):
            success = False
            response_text = requests.post(self.api_url, json=data, headers=headers).text
            response_text = json.loads(response_text)["generated_text"]

            if method == "winrate":
                regex_str = rf'\nMore {dimension["name"]} response: ([1-2])'
            else:
                regex_str = rf'\n{dimension["name"].capitalize()} Score: ([12345])'

            match = re.search(regex_str, response_text)
            if match:
                winner = match.group(1)
                explanation = response_text[:match.start()].rstrip("\n").lstrip("\n")
                success = True
                break
        if not success:
            winner, explanation = -1, "Error in syntax retrieval"

        return {dimension["name"]: float(winner), "id": i, "explanation": explanation}


    def score(self, output_list, src_list, context_list, dimension, batch_size=16):
        # Builds a prompt and submits it in distributed fashion to the local LLM API.
        # Then extracts score and explanation from the response.
        winexpls = []

        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = [executor.submit(self.build_and_submit_prompt, i, output_list, src_list, context_list, dimension)
                        for i, output in enumerate(output_list)]
            for future in tqdm(as_completed(futures), total=len(output_list)):
                winexpl = future.result()
                winexpls.append(winexpl)

        winexpls = sorted(winexpls, key=lambda x: x["id"])
        return winexpls