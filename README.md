# Text-Summarization Evaluation Tools
Migrated from [text summary evaluation](https://github.com/innerNULL/mia/tree/main/bin/evaluation/text_summarisation) on 2024-05-30.

## All-in-One Evaluation Program
First prepare you Pyhton runtime:
```shell
conda create -p ./_venv python=3.10
conda activate ./_venv
conda deactivate

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
