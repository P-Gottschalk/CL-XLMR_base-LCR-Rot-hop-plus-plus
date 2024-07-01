from typing import Optional

import torch
from transformers import XLMRobertaTokenizer, XLMRobertaModel

print()

class EmbeddingsLayerXLMR:

    """
    str training determines which RoBERTa base you use
    Opt. 1: "xlm-roberta-base"
    Opt. 2: "xlm-roberta-large"
    """

    def __init__(self, training: str, device = torch.device('cpu')):
        super().__init__()
        
        self.device = device
        self.tokenizer: XLMRobertaTokenizer = XLMRobertaTokenizer.from_pretrained(training)
        self.model: XLMRobertaModel = XLMRobertaModel.from_pretrained(training, output_hidden_states = True)
        self.model.eval()
    
    def forward(self, sentence: str, target_start: int, target_end: int) -> tuple[
        torch.Tensor, tuple[int, int], Optional[torch.Tensor]
    ]:
        sentence = f"[CLS] {sentence} [SEP]"
        target_start += 6
        target_end += 6


        left_str = self.tokenizer.tokenize(sentence[0:target_start])
        target_str = self.tokenizer.tokenize(sentence[target_start:target_end])
        target_index_start = len(left_str) - 1
        target_index_end = target_index_start + len(target_str)

        tokens = self.tokenizer(sentence, return_tensors="pt")
        
        with torch.no_grad():
            results = self.model(**tokens)

        last_hidden_states = results.hidden_states
        
        last_layers = last_hidden_states[-4:]
        unrounded_embeddings = torch.stack(last_layers).sum(dim=0) / 4

        n_digits = 8
        embeddings = torch.round(unrounded_embeddings * 10 ** n_digits) / (10 ** n_digits)

        embeddings = embeddings[0][1:-1]

        return embeddings, (target_index_start, target_index_end), None


