# Text-Summarization Evaluation Tools
Migrated from [text summary evaluation](https://github.com/innerNULL/mia/tree/main/bin/evaluation/text_summarisation) on 2024-05-30.


This is a simply using all-in-one evaluation program for text-summarization result saved in JSON lines.

## Usage
First prepare you Pyhton runtime:
```shell
conda create -p ./_venv python=3.10
conda activate ./_venv
# conda deactivate

# or 

python3 -m venv ./_venv --copies
source ./_venv/bin/active
python -m pip install -r requirements.txt
# desctivate
```

Then can try with 
```shell
python ./eval_all_in_one_standalone.py ./eval_all_in_one_standalone.json
```
And it will out put JSON lines file, each line contains reference, candidate and metrics name/value.
Currently 4 metrics are supported:
* ROUGE
* METEOR
* [BERTScore](https://arxiv.org/abs/1904.09675): Implemented by myself, not as fast as open-sourced solution, but handled most corner cases.
* Average Sentence Similarity Based on Decode-Only LM, this is for debug purpose only.

The results looks like:
```
================= meteor =================
{
  "meteor": 0.995911574074074
}
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 131.47it/s]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  4.93it/s]
================= bertscore =================
{
  "r_bert": 1.0000000408446754,
  "p_bert": 1.0000000408446754,
  "f_score": 1.0000000408446754
}
./_text_summarization_evaluated_samples.bertscore.jsonl
================= rouge =================
{
  "rouge1_fmeasure": 1.0,
  "rouge1_precision": 1.0,
  "rouge1_recall": 1.0,
  "rouge2_fmeasure": 1.0,
  "rouge2_precision": 1.0,
  "rouge2_recall": 1.0,
  "rougeL_fmeasure": 1.0,
  "rougeL_precision": 1.0,
  "rougeL_recall": 1.0,
  "rougeLsum_fmeasure": 1.0,
  "rougeLsum_precision": 1.0,
  "rougeLsum_recall": 1.0
}
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 29.65it/s]
================= avg_cos_sim =================
{
  "cos_sim": 0.9999999642372132
}
```
