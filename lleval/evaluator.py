from typing import Any
import json
import numpy as np
from .utils.utilities import print_scores

class PromptTemplate:
    def __init__(self, prompt_config_file) -> None:
        """ Load the prompt template from the config file which is in json"""
        with open(prompt_config_file) as f:
            self.prompt_config = json.load(f)

        self.init_prompt = self.prompt_config["init_prompt"]
        self.task_prompt = self.prompt_config["task_prompt"]
        self.context_prompt = self.prompt_config["context_prompt"]
        self.candidate_prompt = self.prompt_config["candidate_prompt"]
        self.eval_prompt = self.prompt_config["eval_prompt"]
        self.post_prompt = self.prompt_config["post_prompt"]

    def format_context(self, dimension, turn_history: list, knowledge_context: list):
        # Turn history is alternating between user and system. Concatenate this list followed by the knowledge context
        # The last element of the turn history is always the system response
        concatenated_context = ""
        if dimension["use_history"]:
            n_turns = dimension["history_turns"]
            total_turns = len(turn_history)
            # only consider last n_turns
            if total_turns > n_turns and n_turns > 0:
                turn_history = turn_history[total_turns - n_turns:]

            concatenated_context = "## Conversation\n"
            for i in range(len(turn_history)):
                if len(turn_history) % 2 == 0:
                    if i % 2 == 0:
                        concatenated_context += "Assistant: " + turn_history[i] + " "
                    else:
                        concatenated_context += "User: " + turn_history[i] + " "
                else:
                    if i % 2 == 0:
                        concatenated_context += "User: " + turn_history[i] + " "
                    else:
                        concatenated_context += "Assistant: " + turn_history[i] + " "
        if dimension["use_knowledge"]:
            concatenated_context += "\n\n## Provided Knowledge\n"
            for i in range(len(knowledge_context)):
                concatenated_context += knowledge_context[i] + " "
        return concatenated_context

    def get_prompt(self, dimension, output, turn_history: list, knowledge_context: list, task_description=""):
        prompt = self.init_prompt + self.task_prompt + " " + dimension["description"] + " " + task_description + self.context_prompt.format(
            self.format_context(dimension, turn_history, knowledge_context)) \
        + "\n\n" + self.candidate_prompt.format(output) \
        + "\n\n" + self.eval_prompt.format(dimension["name"].capitalize()) + self.post_prompt
        return prompt

class DialogEvaluator:
    def __init__(self, prompt_scorer, dimension_definitions_file, max_length=1024, device='cuda:0', cache_dir=None):
        """ Set up evaluator for dialogues """
        self.scorer = prompt_scorer
        self.task = 'dialogue'
        # Load the dimension map from dimension_definitions_file

        self.dimension_map = {}
        if dimension_definitions_file is not None:
            with open(dimension_definitions_file) as f:
                self.dimension_map = json.load(f)
        else:
            raise ValueError("Dimension definitions file is not provided")
        # Get the list of available dimensions by all keys in the dimension map
        self.dimensions = self.dimension_map.keys()

    def evaluate(self, data, dims=None, overall=True, print_result=False, print_expls=False):
        """
            Get the scores of all the given dimensions

            dims: A list of dimensions to be evaluated. If dims is None, DialogEvaluator will evaluate
                  five dimensions: naturalness, coherence, engagingness, groundedness and understandability.

            overall: indicates whether the overall score is to be calculated.
                     Overall score can be customized to a combination of scores based on different
                     dimensions. The default here is the average score of all the given dimensions.

            expls: whether to print the explanation of the score on the screen for each dimension
                     
            print_result: whether to print the average score of each dimension on the screen
        """
        n_data = len(data)
        eval_scores = [{} for _ in range(n_data)]
        eval_expls = [{} for _ in range(n_data)]

        if dims == None:
            eval_dims = self.dimensions
        else:
            assert isinstance(dims, list)
            eval_dims = dims

        for dim in eval_dims:
            dimension=self.dimension_map[dim]
            print('Evaluating {} of {} samples !!!'.format(dim, n_data))
            
            src_list, output_list, context_list = [], [], []
            for i in range(n_data):
                src_list.append(data[i]['source'])
                output_list.append(data[i]['system_output'])
                context_list.append(data[i]['context'])

            score = self.scorer.score(output_list, src_list, context_list, dimension=dimension)
            
            # Numeric scores
            for i in range(n_data):
                eval_scores[i][dim] = score[i][dim]
                eval_expls[i][dim] = score[i]['explanation']

        # Customize your overall score here.
        if overall == True:
            for i in range(n_data):
                eval_scores[i]['overall'] = np.mean(list(eval_scores[i].values()))

        if print_result == True:
            print_scores(eval_scores)

        if print_expls == True:
            for i in range(n_data):
                print('Explanation of the score for sample {}:'.format(i))
                for dim in eval_dims:
                    print('{}: {}'.format(dim, eval_expls[i][dim]))

        return eval_scores, eval_expls
