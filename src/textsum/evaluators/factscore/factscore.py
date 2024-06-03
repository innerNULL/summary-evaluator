# -*- coding: utf-8 -*-
# file: factscore.py
# date: 2024-06-03


import pdb
import sys
import os
import traceback
import re
import json
import torch
import evaluate
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from torch import Tensor
from torch.nn import Module
from transformers import AutoModel, AutoTokenizer
from torchmetrics.text.rouge import ROUGEScore
from langchain_community.chat_models import ChatOllama
from langchain_community.llms import Ollama
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.base import RunnableSequence
from langchain_core.language_models import BaseLanguageModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_chroma import Chroma
from transformers import PreTrainedModel
from pydantic.v1.main import ModelMetaclass


PROMPT_TEMP_ATOMIC_FACT_GENERATION: str = (
    "Please breakdown the following sentence into independent facts: {text}\n"
    "\n"
    "You MUST only return a parsable JSON list `[]` without any other text, "
    "each fact should be one element in the list."
)


LC_CHAT_TEMP_ATOMIC_FACT_GENERATION = ChatPromptTemplate.from_messages(
    [("system", PROMPT_TEMP_ATOMIC_FACT_GENERATION)]
)


PROMPT_TEMP_ATOMIC_FACT_NO_CTX_VALIDATION: str = (
    "{atomic_fact} True or False?"
)


LC_CHAT_TEMP_ATOMIC_FACT_NO_CTX_VALIDATION = ChatPromptTemplate.from_messages(
    [("system", PROMPT_TEMP_ATOMIC_FACT_NO_CTX_VALIDATION)]
)


PROMPT_TEMP_ATOMIC_FACT_RETRIVE_VALIDATION: str = (
    "Context:\n"
    "{context}\n"
    "\n"
    "Regard to above contexts, is {atomic_fact} True or False? "
    "You MUST only return True or False."
)


LC_CHAT_TEMP_ATOMIC_FACT_RETRIVE_VALIDATION = ChatPromptTemplate.from_messages(
    [("system", PROMPT_TEMP_ATOMIC_FACT_RETRIVE_VALIDATION)]
)


def split_sentences(text):
    # Define the regular expression pattern for splitting
    pattern = r'[.。？?\n\n]'
    
    # Split the text using the pattern
    sentences = re.split(pattern, text)
    
    # Remove any leading/trailing whitespace from each sentence
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    
    return sentences


# From https://github.com/langchain-ai/langchain/pull/20689
class HuggingFaceEncoderEmbeddings(BaseModel, Embeddings):
    """HuggingFace encoder-only embedding models.
    
    To use, you must have torch and transformers be installed, 
    some examples are BERT, distil-BERT.
    Test case can be run by `python -m pytest tests/unit_tests/embeddings/test_huggingface.py`
    Example:
        .. code-block:: python
        from langchain_community.embeddings.huggingface import HuggingFaceEncoderEmbeddings
        model_name = "distilbert/distilbert-base-uncased" 
        tokenizer_name = "distilbert/distilbert-base-uncased"
        model_kwargs = {}
        tokenizer_kwargs = {"max_length": 768, "add_special_tokens": False}
        device = "cpu"
        batch_size = 2
        use_cls_embedding=False # This means only use CLS embedding as output.
        embedding = HuggingFaceEncoderEmbeddings(
            model_name=model_name, 
            tokenizer_name=tokenizer_name, 
            device=device, 
            batch_size=batch_size,
            use_cls_embedding=use_cls_embedding, 
            model_kwargs=model_kwargs, 
            tokenizer_kwargs=tokenizer_kwargs
        )
    """
    model_name: str 
    tokenizer_name: str
    device: str = "cpu"
    batch_size: int = 4
    use_cls_embedding: bool = False
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    tokenizer_kwargs: Dict[str, Any] = Field(default_factory=dict)

    client: Any
    tokenizer: Any
    max_length: int = 512
    add_special_tokens: bool = False
    truncation: bool = True

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
        except ImportError as e:
            raise ImportError(
                "Can not import torch and transformers successfully"
            ) from e

        self.client = AutoModel\
            .from_pretrained(self.model_name, **self.model_kwargs)\
            .to(self.device)
        self.tokenizer = AutoTokenizer\
            .from_pretrained(self.tokenizer_name, **self.tokenizer_kwargs)

        if "max_length" in self.tokenizer_kwargs:
            self.max_length = self.tokenizer_kwargs["max_length"]
        if "add_special_tokens" in self.tokenizer_kwargs:
            self.add_special_tokens = self.tokenizer_kwargs["add_special_tokens"]
        if "truncation" in self.tokenizer_kwargs:
            self.truncation = self.tokenizer_kwargs["truncation"]

        torch.set_grad_enabled(False)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        out_embds: List[List[float]] = []
        batch: List[str] = []
        for i, text in enumerate(texts):
            batch.append(text)
            if len(batch) == self.batch_size or i == len(texts) - 1:
                # transformer.BatchEncoding
                tokens = self.tokenizer(
                    batch, 
                    padding=True, 
                    add_special_tokens=self.add_special_tokens, 
                    max_length=self.max_length, 
                    truncation=self.truncation, 
                    return_tensors='pt'
                ).to(self.device)

                self.client.eval()
                # torch.Tensor
                embd_vecs = None
                if self.use_cls_embedding:
                    embd_vecs = self.client(**tokens)["last_hidden_state"][:, 0, :]
                else:
                    # torch.Tensor
                    valid_length = tokens.attention_mask.sum(dim=1)
                    # Keep each embedding at least has valid length larger or 
                    # equal with 1
                    valid_length[valid_length == 0] = 1
                    valid_length = valid_length.reshape(valid_length.shape[0], -1)

                    # torch.Tensor
                    hidden_states = \
                        self.client(**tokens)["last_hidden_state"]
                    hidden_states = hidden_states * tokens.attention_mask.unsqueeze(2)

                    embd_vecs = hidden_states.sum(dim=1) / valid_length

                out_embds.extend(embd_vecs.tolist())
                batch = []
        return out_embds

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class FactScoreWithLlmServer:
    def __init__(self):
        self.model: Optional[BaseChatModel] = None
        self.encoder: Optional[HuggingFaceEncoderEmbeddings] = None

    def init_with_llm_and_encoder(self, 
        llm_cli: BaseChatModel, 
        encoder: HuggingFaceEncoderEmbeddings
    ) -> None:
        self.model = llm_cli
        self.encoder = encoder
        self.facts_generator = LC_CHAT_TEMP_ATOMIC_FACT_GENERATION \
            | self.model \
            | StrOutputParser()
        self.fact_no_ctx_validator = LC_CHAT_TEMP_ATOMIC_FACT_NO_CTX_VALIDATION \
            | self.model \
            | StrOutputParser()
        self.fact_retrive_validator = LC_CHAT_TEMP_ATOMIC_FACT_RETRIVE_VALIDATION \
            | self.model \
            | StrOutputParser()

    def init_with_llm_and_encoder_configs(self,
        llm: str="llama3:8b",
        server_type: str="ollama", 
        url: str="http://localhost:11434",
        encoder_name_or_path: str="emilyalsentzer/Bio_ClinicalBERT", 
        tokenizer_name_or_path: Optional[str]=None
    ) -> None:
        if tokenizer_name_or_path is None:
            tokenizer_name_or_path = encoder_name_or_path

        llm_cli_cls: Optional[ModelMetaclass] = None
        llm_cli: Optional[BaseChatModel] = None
        if server_type == "ollama":
            llm_cli_cls = ChatOllama
            llm_cli = llm_cli_cls(
                model=llm, base_url=url, temperature=0, top_k=1
            )
        encoder: HuggingFaceEncoderEmbeddings = HuggingFaceEncoderEmbeddings(
            model_name=encoder_name_or_path, 
            tokenizer_name=tokenizer_name_or_path
        )
        self.init_with_llm_and_encoder(llm_cli, encoder)

    def generate_facts(self, input_text: str) -> List[str]:
        output: str = self.facts_generator.invoke({"text": input_text})
        try:
            return json.loads(output)
        except Exception as e:
            print(output)
            raise e

    def build_knowledge_source(self, knowledges: Union[str, List[str]]):
        if isinstance(knowledges, str):
            knowledges = split_sentences(knowledges)
        knowledge_docs: List[Document] = [
            Document(page_content=x, metadata={}) for x in knowledges
        ]
        vectorstore: Chroma = Chroma.from_documents(
            knowledge_docs, embedding=self.encoder, 
            collection_metadata={"hnsw:space": "cosine"}
        )
        return vectorstore

    def generate_factscore(self, 
        input_text: str, 
        knowledges: Union[str, List[str]], 
        mode: str="retrive"
    ):
        facts: List[str] = self.generate_facts(input_text)
        knowledges: Chroma = self.build_knowledge_source(knowledges)
        retriever = knowledges.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 1},
        )
        val_results: List[int] = []
        for fact in facts:
            val_result: str = None
            if mode == "no_ctx":
                val_result = self.fact_no_ctx_validator.invoke({"atomic_fact": fact})
            elif mode == "retrive":
                retrived_docs: List[Documents] = retriever.batch([fact])[0]
                val_result = self.fact_retrive_validator.invoke(
                    {"atomic_fact": fact, "context": retrived_docs[0].page_content}
                )
            else:
                raise "Invalid `mode`"
            if val_result.lower() == "true":
                val_results.append(1)
            elif val_result.lower() == "false":
                val_results.append(0)
        return sum(val_results) / (len(val_results) + 0.0001)


if __name__ == "__main__":
    test_data = """
    It is uncommon for hospital staff to be required to take forensic specimens. Details of all specimens obtained should appear in the medico-legal report There should be clear notation as to the site from which the specimens derived, the way they were labelled, details of handling and the reason for obtaining that specimen (for example bacteriology for comparison purposes). Comments should also be made regarding the time and date of transfer of specimens to the care of another person. This ensures that continuity of evidence can be proven later in court. The report should refer to any photographs taken and the text should clearly identify each photograph.
    """
    sentences = split_sentences(test_data)
    factscore = FactScoreWithLlmServer()
    factscore.init_with_llm_and_encoder_configs()
    a = factscore.generate_factscore(".".join(sentences[:3]), test_data)
    pdb.set_trace()
