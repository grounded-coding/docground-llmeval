import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

class LLEvaluator:
    def __init__(self, model_name_or_path, max_length=1024, device='cuda:0', cache_dir=None):
        """ Set up model """
        self.device = device
        self.max_length = max_length

        self.model.eval()
        self.model.to(device)

    # TODO Integrate Llama2Local as the default scorer model here

    def score(self, inputs, batch_size=8):
        """
            Get scores for the given samples.
            final_score = postive_score / (postive_score + negative_score)
        """
        tgts = ["No" for _ in range(len(inputs))]

        pos_score_list, neg_score_list = [], []
        for i in tqdm(range(0, len(inputs), batch_size)):
            src_list = inputs[i: i + batch_size]
            tgt_list = tgts[i: i + batch_size]
            try:
                with torch.no_grad():
                    encoded_src = self.tokenizer(
                        src_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    encoded_tgt = self.tokenizer(
                        tgt_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )

                    src_tokens = encoded_src['input_ids'].to(self.device)
                    src_mask = encoded_src['attention_mask'].to(self.device)

                    tgt_tokens = encoded_tgt['input_ids'].to(self.device)[:, 0].unsqueeze(-1)

                    output = self.model(
                        input_ids=src_tokens,
                        attention_mask=src_mask,
                        labels = tgt_tokens
                    )
                    logits = output.logits.view(-1, self.model.config.vocab_size)
            
                    pos_score = self.softmax(logits)[:, self.pos_id] # Yes
                    neg_score = self.softmax(logits)[:, self.neg_id] # No

                    cur_pos_score = [x.item() for x in pos_score]
                    cur_neg_score = [x.item() for x in neg_score]
                    pos_score_list += cur_pos_score
                    neg_score_list += cur_neg_score

            except RuntimeError:
                print(f'source: {src_list}')
                print(f'target: {tgt_list}')
                exit(0)
        
        score_list = []
        for i in range(len(pos_score_list)):
            score_list.append(pos_score_list[i] / (pos_score_list[i] + neg_score_list[i]))
            
        return score_list