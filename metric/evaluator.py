from llama2_local import Llama2Local
from metric.scorer import LLEvaluator
import numpy as np
from utils import add_prompt_instructions, print_scores

# Define abstract class prompt evaluator which should define methods to build and pass a prompt for evaluation to the local model,
# outputs a score and an explanation for a given input dialog history, knowledge string, and two candidate responses.
class PromptEvaluator:
    def build_prompt(self, metric, context, response_1, response_2=None, method="winrate"):
        raise NotImplementedError()
    
    def submit_prompt(self, prompt, method, metric):
        raise NotImplementedError()

class DialogEvaluator:
    def __init__(self, max_length=1024, device='cuda:0', cache_dir=None):
        """ Set up evaluator for dialogues """
        llama2_local = Llama2Local(api_url='http://gpu-19.apptek.local:8080/generate')
        self.scorer = LLEvaluator(model_prompter=llama2_local, 
                                   max_length=max_length, 
                                   device=device, cache_dir=cache_dir)
        self.task = 'dialogue'
        self.dimensions = ['accuracy', 'appropriateness']

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
            
            # Calculate turn-level score for other dimensions
            if dim in ['accuracy', 'appropriateness']:
                src_list, output_list, context_list = [], [], []
                for i in range(n_data):
                    src_list.append(data[i]['source'])
                    output_list.append(data[i]['system_output'])
                    context_list.append(data[i]['context'])
                input_list = add_prompt_instructions(dimension=dim, output=output_list, 
                                          src=src_list, context=context_list, task=self.task)
                score = self.scorer.score(input_list)

            # Please customize other dimensions here for summarization
            else:
                raise NotImplementedError('The input format for this dimension is still undefined. \
                                           Please customize it first.')
            
            for i in range(n_data):
                eval_scores[i][dim] = score[i]

        # Customize your overall score here.
        if overall == True:
            for i in range(n_data):
                eval_scores[i]['overall'] = np.mean(list(eval_scores[i].values()))

        if print_result == True:
            print_scores(eval_scores)

        return eval_scores