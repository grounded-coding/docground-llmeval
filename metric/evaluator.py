import numpy as np
from utils import add_prompt_instructions, print_scores
import random

class DialogEvaluator:
    def __init__(self, prompt_scorer, max_length=1024, device='cuda:0', cache_dir=None):
        """ Set up evaluator for dialogues """
        self.scorer = prompt_scorer
        self.task = 'dialogue'
        self.dimension_map = {
            'appropriate': {
                'description': 'Rate how appropriate this response is. An appropriate response is naturally connected to the previous turns and helpful for the user\'s concern. Do not consider factual correctness.',
            },
            'accurate': {
                'description': 'Rate how accurate this response is. An accurate response is factually correct and consistent with the knowledge in the context.',
                'data_specific_task_description': 'Context can be reviews from customers or FAQs. FAQs start after token :F: and each new review starts after token :R:.'
            },
        }
        self.dimension = ["appropriate", "accurate"]
        self.init_prompt = "[INST] <<SYS>>"
        self.task_prompt = "For the following multi-turn conversation between User and Assistant, you will be given a potential response for the next turn."
        self.context_prompt = "<</SYS>> {}"
        self.candidate_prompt = "\n\n## Response\n{}\n\n## "
        self.eval_prompt = " Task\nFIRST provide a one-sentence explanation of your rating. SECOND, state only state only the score on a scale of 1 to 5. Follow the template.\n\n## Template\nExplanation: <one-sentence explanation>\n{} Score: <1-5>"
        self.post_prompt = "[/INST]Explanation:"

    def build_dim_prompt(self, dimension, context, response_1, response_2):
        # Concatenate init, task, selected dimensions and post prompts
        prompt = self.init_prompt + self.task_prompt + self.selected_dimensions[dimension]['description'] + \
                    self.selected_dimensions[dimension]['data_specific_task_description'] + self.post_prompt
        # Now format the prompt with the context, response_1 and response_2
        prompt = prompt.format(context, response_1, response_2, dimension.capitalize())
        return prompt

    def evaluate(self, data, dims=None, overall=True, print_result=False):
        """
            Get the scores of all the given dimensions

            dims: A list of dimensions to be evaluated. If dims is None, DialogEvaluator will evaluate
                  five dimensions: naturalness, coherence, engagingness, groundedness and understandability.

            overall: indicates whether the overall score is to be calculated.
                     Overall score can be customized to a combination of scores based on different
                     dimensions. The default here is the average score of all the given dimensions.

            print_result: whether to print the average score of each dimension on the screen
        """
        n_data = len(data)
        eval_scores = [{} for _ in range(n_data)]

        if dims == None:
            eval_dims = self.dimensions
        else:
            assert isinstance(dims, list)
            eval_dims = dims

        for dim in eval_dims:
            print('Evaluating {} of {} samples !!!'.format(dim, n_data))
            
            src_list, output_list, context_list = [], [], []
            for i in range(n_data):
                src_list.append(data[i]['source'])
                output_list.append(data[i]['system_output'])
                context_list.append(data[i]['context'])

            score = self.scorer.score(output_list, src_list, context_list, dimension=dim)
            
            for i in range(n_data):
                eval_scores[i][dim] = score[i]

        # Customize your overall score here.
        if overall == True:
            for i in range(n_data):
                eval_scores[i]['overall'] = np.mean(list(eval_scores[i].values()))

        if print_result == True:
            print_scores(eval_scores)

        return eval_scores