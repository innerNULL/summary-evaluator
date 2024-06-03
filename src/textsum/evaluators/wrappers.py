# -*- coding: utf-8 -*-
# file: wrappers.py
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

from .interfaces import BaseMetric


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

