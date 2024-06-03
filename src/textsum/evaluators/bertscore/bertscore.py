# -*- coding: utf-8 -*-
# file: bertscore.py
# date: 2024-06-03


import pdb
import sys
import os
import traceback
import json
import torch
import evaluate
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Any, Optional, Set, Tuple
from torch import Tensor
from torch.nn import Module
from transformers import AutoModel, AutoTokenizer
from torchmetrics.text.rouge import ROUGEScore

from ..interfaces import BaseMetric


def sentence_cos_sim(
    encoder: Module, 
    tokenizer: Any, 
    target_text: str, 
    output_text: str,
    device: torch.device = torch.device("cpu"),
    use_cls_embedding: bool=False
) -> float:
    target_tokens: Tensor = tokenizer.encode_plus(
        target_text, add_special_tokens=True, return_tensors='pt'
    ).to(device)
    output_tokens: Tensor = tokenizer.encode_plus(
        output_text, add_special_tokens=True, return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        target_embd: Tensor = None
        output_embd: Tensor = None
        if use_cls_embedding:
            target_embd = model(**target_tokens)["last_hidden_state"][0, 0, :]
            output_embd = model(**output_tokens)["last_hidden_state"][0, 0, :]
        else:
            target_embd = torch.mean(
                encoder(**target_tokens)["last_hidden_state"][:, 1:, :], dim=1
            ).squeeze()
            output_embd = torch.mean(
                encoder(**output_tokens)["last_hidden_state"][:, 1:, :], dim=1
            ).squeeze()
        
        cos_sim: float = torch.cosine_similarity(
            target_embd.reshape(1, -1), output_embd.reshape(1, -1)
        ).cpu().tolist()[0]
    return cos_sim



class AvgCosSimilarity(BaseMetric):
    def __init__(self, 
        encoder: Optional[Module],
        tokenizer: Optional[Any],
        target_texts: Optional[List[str]],
        device: str="cuda:1"
    ):
        super().__init__(encoder, tokenizer, target_texts, device)

    def run(
        self, target_texts: List[str], pred_texts: List[str]
    ) -> Tuple[ Dict, Optional[List[Dict]] ]:
        assert(len(target_texts) == len(pred_texts))
        embeddings: List[Dict] = []
        for i in tqdm(range(len(target_texts))):
            try:
                target_text: str = target_texts[i]
                output_text: str = pred_texts[i] 
                cos_sim: float = sentence_cos_sim(
                    self.encoder, self.tokenizer, target_text, output_text, 
                    device=self.device,
                    use_cls_embedding=False
                )
                embedding: Dict = {
                    "target_text": target_text,
                    "output_text": output_text,
                    #"target_embd": target_embd.cpu().tolist(), 
                    #"output_embd": output_embd.cpu().tolist(),
                    "cos_sim": cos_sim
                }
                embeddings.append(embedding)
            except Exception as e:
                print("Failed on some marginal cases, following are error messages")
                print(e)
                print(traceback.format_exc())

        cos_sims: List[float] = [x["cos_sim"] for x in embeddings]
        return {"cos_sim": sum(cos_sims)  / len(cos_sims)}, None
     

class BertScore(BaseMetric):
    def __init__(self,
        encoder: Optional[Module],
        tokenizer: Optional[Any],
        target_texts: Optional[List[str]],
        device: str="cuda:1"
    ):
        super().__init__(encoder, tokenizer, target_texts, device)
        self.build_token_id_df()

    def build_token_id_df(self) -> Dict[int, float]:
        for target_text in tqdm(self.target_texts):
            token_ids: List[int] = self.tokenizer.encode_plus(
                target_text, add_special_tokens=True
            )["input_ids"]
            for token_id in set(token_ids):
                if token_id not in self.token_id_df:
                    self.token_id_df[token_id] = 0
                self.token_id_df[token_id] += 1

    def get_idf(self, token_id: int) -> float:
        df: int = self.token_id_df.get(token_id, 0) 
        assert(df <= len(self.target_texts))

        # Refer to paper https://arxiv.org/abs/1904.09675,
        # using "plus one" smooth here
        return -np.log(
            (df + 1) / (len(self.target_texts) + 1)
        )

    def get_idf_weights(self, ids: List[int]) -> Tensor:
        idf_vals: List[float] = [self.get_idf(i) for i in ids]
        idf_vals = [x if x > 0.0 else 0.0 for x in idf_vals]
        
        idf_sum: float = sum(idf_vals)
        idf_sum: float = 0.00001 if idf_sum == 0 else idf_sum

        idf_weights: List[float] = [x / idf_sum for x in idf_vals]
        return torch.tensor(idf_weights).to(self.device) 

    def _run(self, target_texts: List[str], pred_texts: List[str]) -> Dict:
        assert(len(target_texts) == len(pred_texts))
        recorder: Dict[str, List[float]] = {
            "r_bert": [], "p_bert": [], "f_score": [], 
            "candidate": [], "reference": []
        }
        for i in tqdm(range(len(target_texts))):
            try:
                target_text: str = target_texts[i]
                pred_text: str = pred_texts[i]
                target_tokens: Tensor = self.tokenizer.encode_plus(
                    target_text, add_special_tokens=True, return_tensors='pt',
                    truncation=True, 
                    #padding='max_length', 
                    max_length=512
                ).to(self.device)
                pred_tokens: Tensor = self.tokenizer.encode_plus(
                    pred_text, add_special_tokens=True, return_tensors='pt',
                    truncation=True, 
                    #padding='max_length', 
                    max_length=512
                ).to(self.device)
                
                target_idf_weights: Tensor = self.get_idf_weights(
                    target_tokens.input_ids.cpu().tolist()[0][1:]
                )
                pred_idf_weights: Tensor = self.get_idf_weights(
                    pred_tokens.input_ids.cpu().tolist()[0][1:]
                )

                with torch.no_grad():
                    target_embds: Tensor = \
                        self.encoder(**target_tokens)["last_hidden_state"][:, 1:, :].squeeze()
                    pred_embds: Tensor = \
                        self.encoder(**pred_tokens)["last_hidden_state"][:, 1:, :].squeeze()
                    # This handle case when generated a single or empty string.
                    pred_embds = pred_embds.reshape(-1, pred_embds.shape[-1])
                    
                    target_embds = target_embds / (target_embds * target_embds)\
                        .sum(dim=1)\
                        .pow(0.5)\
                        .reshape(-1, target_embds.shape[0])\
                        .repeat(target_embds.shape[1], 1).T
                    pred_embds = pred_embds / (pred_embds * pred_embds)\
                        .sum(dim=1)\
                        .pow(0.5)\
                        .reshape(-1, pred_embds.shape[0])\
                        .repeat(pred_embds.shape[1], 1).T
                    
                    # pred token ID num * target token ID num
                    cos_sim: Tensor = target_embds.matmul(pred_embds.T)
                    r_bert: Tensor = (torch.max(cos_sim, dim=1).values * target_idf_weights).sum()
                    p_bert: Tensor = (torch.max(cos_sim, dim=0).values * pred_idf_weights).sum()
                    f_score: Tensor = 2.0 * (r_bert * p_bert) / (r_bert + p_bert)
                
                recorder["r_bert"].append(r_bert.cpu().tolist())
                recorder["p_bert"].append(p_bert.cpu().tolist())
                recorder["f_score"].append(f_score.cpu().tolist()) 
                recorder["candidate"].append(pred_text)
                recorder["reference"].append(target_text)
            except Exception as e:
                print("Failed on some marginal cases, following are error messages")
                print(e)
                print(traceback.format_exc())
                pdb.set_trace()
        return recorder

    def run(
        self, target_texts: List[str], pred_texts: List[str]
    ) -> Tuple[ Dict, Optional[List[Dict]] ]:
        bert_scores: Dict = self._run(target_texts, pred_texts)
        assert(len(bert_scores["candidate"]) == len(bert_scores["reference"]))
        assert(len(bert_scores["f_score"]) == len(bert_scores["reference"]))
        assert(len(bert_scores["r_bert"]) == len(bert_scores["reference"]))
        assert(len(bert_scores["p_bert"]) == len(bert_scores["reference"]))

        bertscore_metrics: Dict[str, float] = {
            "r_bert": sum(bert_scores["r_bert"]) / len(bert_scores["r_bert"]), 
            "p_bert": sum(bert_scores["p_bert"]) / len(bert_scores["p_bert"]),
            "f_score": sum(bert_scores["f_score"]) / len(bert_scores["f_score"])
        }
        sample_metris: List[str] = []
        for i in range(len(bert_scores["f_score"])):
            record: Dict = {
              "candidate": bert_scores["candidate"][i],
              "reference": bert_scores["reference"][i],
              "r_bert": bert_scores["r_bert"][i],
              "p_bert": bert_scores["p_bert"][i],
              "f_score": bert_scores["f_score"][i]
            }
            sample_metris.append(record)
        return bertscore_metrics, sample_metris
