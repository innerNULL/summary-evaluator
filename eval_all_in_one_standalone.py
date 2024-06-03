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

from src.textsum.evaluators import Rouge
from src.textsum.evaluators import Meteor
from src.textsum.evaluators import AvgCosSimilarity
from src.textsum.evaluators import BertScore


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
