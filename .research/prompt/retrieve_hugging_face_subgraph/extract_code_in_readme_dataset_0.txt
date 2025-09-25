
Input:
From the Hugging Face README provided in “# README,” extract and output only the Python code required for execution. Do not output any other information. In particular, if no implementation method is described, output an empty string.

# README
---
size_categories:
- 1K<n<10K
task_categories:
- graph-ml
tags:
- art
license: cc
---

## Dataset Information

| # Nodes | # Edges | # Features |
|:-------:|:-------:|:----------:|
|  2,277  |  36,101 |      2,325    |

## Usage

```python
from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="SauravMaheshkar/pareto-chameleon", filename="processed/chameleon.bin", local_dir="./data/", repo_type="dataset")

dataset, _ = dgl.load_graphs("./data/processed/chameleon.bin")
```

Thank you [@severo](https://huggingface.co/severo) for helping me [figure out the usage](https://discuss.huggingface.co/t/can-i-use-a-pickle-file-with-the-data-files-argument-with-datasets/72189/2?u=sauravmaheshkar).

Pre-processed as per the official codebase of https://arxiv.org/abs/2210.02016

## Citations

```
@article{ju2023multi,
  title={Multi-task Self-supervised Graph Neural Networks Enable Stronger Task Generalization},
  author={Ju, Mingxuan and Zhao, Tong and Wen, Qianlong and Yu, Wenhao and Shah, Neil and Ye, Yanfang and Zhang, Chuxu},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```

```
@article{DBLP:journals/corr/abs-1909-13021,
  author       = {Benedek Rozemberczki and
                  Carl Allen and
                  Rik Sarkar},
  title        = {Multi-scale Attributed Node Embedding},
  journal      = {CoRR},
  volume       = {abs/1909.13021},
  year         = {2019},
  url          = {http://arxiv.org/abs/1909.13021},
  eprinttype    = {arXiv},
  eprint       = {1909.13021},
  timestamp    = {Wed, 02 Oct 2019 13:04:08 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-1909-13021.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
Output:
{
    "extracted_code": "from huggingface_hub import hf_hub_download\n\nhf_hub_download(repo_id=\"SauravMaheshkar/pareto-chameleon\", filename=\"processed/chameleon.bin\", local_dir=\"./data/\", repo_type=\"dataset\")\n\ndataset, _ = dgl.load_graphs(\"./data/processed/chameleon.bin\")"
}
