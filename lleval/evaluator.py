from typing import Any
import numpy as np
from .utils.utilities import print_scores

class PromptTemplate:
    def __init__(self) -> None:
        self.init_prompt = "[INST] <<SYS>> "
        self.task_prompt = "For the following multi-turn conversation between User and Assistant, you will be given a potential response for the next turn."
        self.context_prompt = "<</SYS>> {}"
        self.candidate_prompt = "## Response\n{}"
        self.eval_prompt = "## Task\nFIRST provide a one-sentence explanation of your rating. SECOND, state only state only the score on a scale of 1 to 5. Follow the template.\n\n## Template\nExplanation: <one-sentence explanation>\n{} Score: <1-5>"
        self.post_prompt = "[/INST]Explanation:"

    def format_context(self, dimension, turn_history: list, knowledge_context: list):
        # Turn history is alternating between user and system. Concatenate this list followed by the knowledge context
        concatenated_context = "## Conversation\n"
        for i in range(len(turn_history)):
            if i % 2 == 0:
                concatenated_context += "User: " + turn_history[i] + " "
            else:
                concatenated_context += "Assistant: " + turn_history[i] + " "
        if dimension["name"] == "accurate":
            concatenated_context += "\n\n## Context\n"
            for i in range(len(knowledge_context)):
                concatenated_context += knowledge_context[i] + " "
        return concatenated_context

    def get_prompt(self, dimension, output, turn_history: list, knowledge_context: list, dim_description="", task_description=""):
        prompt = self.init_prompt + self.task_prompt + " " + dim_description + " " + task_description + self.context_prompt.format(
            self.format_context(dimension, turn_history, knowledge_context)) \
        + "\n\n" + self.candidate_prompt.format(output) \
        + "\n\n" + self.eval_prompt.format(dimension["name"].capitalize()) + self.post_prompt
        # print(prompt)
        return prompt

class DialogEvaluator:
    def __init__(self, prompt_scorer, max_length=1024, device='cuda:0', cache_dir=None, dataset_task_description=""):
        """ Set up evaluator for dialogues """
        self.scorer = prompt_scorer
        self.task = 'dialogue'
        self.dimension_map = {
            'appropriate': {
                'description': 'Rate how appropriate this response is. An appropriate response is naturally connected to the previous turns and helpful for the user\'s concern. Do not consider factual correctness.',
                'data_specific_task_description': "",
                'name': 'appropriate'
            },
            'accurate': {
                'description': 'Rate how accurate this response is. An accurate response is factually correct and consistent with the knowledge in the context. An inaccurate response invents facts, which cannot be cited or derived from the context. Carefully assess the score if some parts of the response are correct and other parts are wrong. Do not rate helpfulness or the style of the conversation.',
                'data_specific_task_description': dataset_task_description,
                'name': 'accurate'
            },
        }
        self.dimensions = ["appropriate", "accurate"]

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
