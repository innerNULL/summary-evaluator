# -*- coding: utf-8 -*-
# file: al_all_in_one_standalone.py
# date: 2024-03-20
#
# python ./eval_all_in_one_standalone.py ./eval_all_in_one_standalone.json


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


def print_metrics(vals: Dict, name: str) -> None:
    print("================= %s =================" % name)
    print(json.dumps(vals, indent=2))


def load_data(path_or_name: str) -> List[Dict]:
    out: List[Dict] = []
    if os.path.exists(path_or_name):
        if ".csv" in path_or_name:
            out = pd.read_csv(path_or_name).to_dict(orient="records")
        elif ".jsonl" in path_or_name:
            out = [
                json.loads(x) for x in open(path_or_name, "r").read().split("\n")
                if x not in {""}
            ]
        else:
            raise Exception("Not support %s format" % path_or_name)
    else:
        raise Exception("File %s does not exist" % path_or_name)

    return out


def sentence_cos_sim(
    encoder: Module, tokenizer: Any, target_text: str, output_text: str, 
    use_cls_embedding: bool=False
) -> float:
    target_tokens: Tensor = tokenizer.encode_plus(
        target_text, add_special_tokens=True, return_tensors='pt'
    ).to(torch.device(device))
    output_tokens: Tensor = tokenizer.encode_plus(
        output_text, add_special_tokens=True, return_tensors='pt'
    ).to(torch.device(device))

    with torch.no_grad():
        target_embd: Tensor = None
        output_embd: Tensor = None
        if use_cls_embedding:
            target_embd = model(**target_tokens)["last_hidden_state"][0, 0, :]
            output_embd = model(**output_tokens)["last_hidden_state"][0, 0, :]
        else:
            target_embd = torch.mean(
                model(**target_tokens)["last_hidden_state"][:, 1:, :], dim=1
            ).squeeze()
            output_embd = torch.mean(
                model(**output_tokens)["last_hidden_state"][:, 1:, :], dim=1
            ).squeeze()
        
        cos_sim: float = torch.cosine_similarity(
            target_embd.reshape(1, -1), output_embd.reshape(1, -1)
        ).cpu().tolist()[0]
    return cos_sim


class BaseMetric:
    def __init__(self, 
        encoder: Optional[Module]=None, 
        tokenizer: Optional[Any]=None, 
        target_texts: Optional[List[str]]=None, 
        device: str="cuda:1"
    ):
        self.encoder: Module = encoder
        self.tokenizer: Any = tokenizer
        self.target_texts: List[str] = target_texts
        self.token_id_df: Dict[int, float] = {}
        self.device: torch.device = torch.device(device)

    def run(
        self, target_texts: List[str], pred_texts: List[str]
    ) -> Tuple[ Dict, Optional[List[Dict]] ]:
        return {}, None


class Meteor(BaseMetric):
    def run(
        self, target_texts: List[str], pred_texts: List[str]
    ) -> Tuple[ Dict, Optional[List[Dict]] ]:
        meteor = evaluate.load('meteor')
        results = meteor.compute(predictions=pred_texts, references=target_texts)
        return results, None


class Rouge(BaseMetric):
    def __init__(self, 
        encoder: Optional[Module]=None,
        tokenizer: Optional[Any]=None,
        target_texts: Optional[List[str]]=None,
        device: str="cuda:1"
    ):
        super().__init__(encoder, tokenizer, target_texts, device)

    def run(
        self, target_texts: List[str], pred_texts: List[str]
    ) -> Tuple[ Dict, Optional[List[Dict]] ]:
        rouge: ROUGEScore = ROUGEScore()
        metrics: Dict[str, Tensor] = rouge(pred_texts, target_texts)
        return {k: v.cpu().tolist() for k, v in metrics.items()}, None


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
                    use_cls_embedding=False
                )
                embedding: Dict = {
                    configs["target_text_col"]: target_text,
                    configs["output_text_col"]: output_text,
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
                target_tokens: Tensor = tokenizer.encode_plus(
                    target_text, add_special_tokens=True, return_tensors='pt',
                    truncation=True, 
                    #padding='max_length', 
                    max_length=512
                ).to(self.device)
                pred_tokens: Tensor = tokenizer.encode_plus(
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
                        model(**target_tokens)["last_hidden_state"][:, 1:, :].squeeze()
                    pred_embds: \
                        Tensor = model(**pred_tokens)["last_hidden_state"][:, 1:, :].squeeze()
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


if __name__ == "__main__":
    configs: Dict = json.loads(open(sys.argv[1], "r").read())
    print(configs)
    hf_lm_path_or_name: str = configs["hf_lm_path_or_name"]
    device: str = configs["device"]
    metrics: Set[str] = set(configs["metrics"])

    inf_results: List[Dict] = load_data(configs["data_path_or_name"])
    target_texts: List[str] = [x[configs["target_text_col"]] for x in inf_results]
    pred_texts: List[str] = [
        x[configs["output_text_col"]][:configs["max_output_char_num"]] 
        for x in inf_results
    ]

    model = AutoModel.from_pretrained(hf_lm_path_or_name).to(torch.device(device))
    tokenizer = AutoTokenizer.from_pretrained(hf_lm_path_or_name)
    
    sample_metrics: Optional[List[Dict]] = None
    for metric_name in metrics:
        metrics_val: Dict[str, float] = {}
        if metric_name == "bertscore":
            bertscore: BertScore = BertScore(
                encoder=model, tokenizer=tokenizer, 
                target_texts=[x[configs["target_text_col"]] for x in inf_results],
                device=device
            )
            metrics_val, sample_metrics = bertscore.run(target_texts, pred_texts)
            print_metrics(metrics_val, metric_name)
        if metric_name == "avg_cos_sim":
            avg_cos_sim: AvgCosSimilarity = AvgCosSimilarity(
                encoder=model, tokenizer=tokenizer, target_texts=None, device=device
            )
            metrics_val, sample_metrics = avg_cos_sim.run(target_texts, pred_texts)
            print_metrics(metrics_val, metric_name)
        if metric_name == "rouge":
            rouge: Rouge = Rouge()
            metrics_val, sample_metrics = rouge.run(target_texts, pred_texts)
            print_metrics(metrics_val, metric_name)
        if metric_name == "meteor":
            meteor: Meteor = Meteor()
            metrics_val, sample_metrics = meteor.run(target_texts, pred_texts)
            print_metrics(metrics_val, metric_name)

        if sample_metrics is not None:
           ext: str = configs["output_path"].split(".")[-1]
           file_name: str = ".".join(configs["output_path"].split(".")[:-1])
           curr_path = "{}.{}.{}".format(file_name, metric_name, ext)
           f = open(curr_path, "w") 
           for sample in sample_metrics:
               f.write(json.dumps(sample, ensure_ascii=False) + "\n")
           print(curr_path)
