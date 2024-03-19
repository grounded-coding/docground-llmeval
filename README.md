## Getting Started

The following class is a sample implementation of using a local tgi API

Start using 

`from lleval.evaluator import PromptTemplate, DialogEvaluator
from lleval.scorer import PromptScorer`

Below is an example configuration

```
class LLEval(EvaluationFramework):
    def __init__(self, gen_config="configs/llama2/gen_config.json", dim_definitions="configs/dimension_definitions.json",
                 api_url="http://....local:8080/generate", likert_config="configs/llama2/prompt_likert_config.json", name=None):
        super().__init__(['appropriate', 'accurate', "grounded", "coherent"], name=name)
        self.dim_definitions = dim_definitions
        self.gen_config = gen_config
        self.api_url = api_url
        self.likert_config = likert_config

    def evaluate(self, model_responses, reference_responses, turn_historys, knowledge_contexts, dims):
        data = convert_to_json(output_list=model_responses, src_list=turn_historys, context_list=knowledge_contexts)
        prompt_template = PromptTemplate(self.likert_config)
        llama2local = PromptScorer(api_url=self.api_url, metric_config_file=self.gen_config, prompt_template=prompt_template, num_retries=3)
        evaluator = DialogEvaluator(llama2local, dimension_definitions_file=self.dim_definitions)
        eval_scores, eval_expls = evaluator.evaluate(data, print_result=True, dims=dims)
        merged_scores = []
        for i, score in enumerate(eval_scores):
            for key in eval_expls[i].keys():
                score[key + "_expl"] = eval_expls[i][key]
            merged_scores.append(score)
        return merged_scores
```

which would be run using

`framework_scores = lleval.evaluate(model_responses, reference_responses, turn_historys, knowledge_contexts, self.desired_dimensions)`

on a list of preprocessed lists of strings for the contexts and corresponding model responses.