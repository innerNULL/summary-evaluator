# -*- coding: utf-8 -*-
# file: __init__.py
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
