import requests
import json
import re

from metric.evaluator import PromptEvaluator

class Llama2Local(PromptEvaluator):
    def __init__(self, api_url, metric_config_file, num_retries=3):
        self.api_url = api_url
        self.num_retries = num_retries

        with open(metric_config_file, "r") as conf:
            self.metric_config = json.load(conf)

    def build_context(self, turn_history=None, knowledge_context=None):
        return turn_history + knowledge_context

    def build_prompt(self, metric, context, response_1, response_2=None, method="winrate"):
        if method == "winrate":
            if metric == "appropriate":
                return self.metric_config["data_appropriateness"].format(context, response_1, response_2)
            elif metric == "accurate":
                return self.metric_config["data_accuracy"].format(context, response_1, response_2)
            else:
                raise NotImplementedError()
        else:
            if metric == "appropriate":
                return self.metric_config["data_appropriateness"].format(context, response_1)
            elif metric == "accurate":
                return self.metric_config["data_accuracy"].format(context, response_1)
            else:
                raise NotImplementedError()

    def submit_prompt(self, prompt, method, metric):
        data = {
            "inputs": prompt,
            "parameters": self.metric_config["gen_params"]
        }

        headers = {
            'Content-Type': 'application/json'
        }

        for _ in range(self.num_retries):
            response_text = requests.post(self.api_url, json=data, headers=headers).text
            response_text = json.loads(response_text)["generated_text"]

            if method == "winrate":
                regex_str = rf'\nMore {metric} response: ([1-2])'
            else:
                regex_str = rf'\n{metric.capitalize()} Score: ([12345])'

            match = re.search(regex_str, response_text)
            if match:
                winner = match.group(1)
                explanation = response_text[:match.start()].rstrip("\n").lstrip("\n")
                return winner, explanation

        return -1, "Error in syntax retrieval"

    def prompt_eval(self, metric="appropriate", turn_history=None, knowledge_context=None, response_1=None, response_2=None, method="winrate"):
        context = self.build_context(turn_history, knowledge_context)
        prompt = self.build_prompt(metric, context, response_1, response_2, method)
        return self.submit_prompt(prompt, method, metric)


api_url = 'http://gpu-19.apptek.local:8080/generate'
metric_config_file = "metric_winrate_config.json"
llama2local = Llama2Local(api_url, metric_config_file)

winner, explanation = llama2local.prompt_eval(metric="appropriate", turn_history="Example turn history", knowledge_context="Example knowledge context", response_1="Example response 1", response_2="Example response 2")
print(winner, explanation)