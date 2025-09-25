
Input:
From the Hugging Face README provided in “# README,” extract and output only the Python code required for execution. Do not output any other information. In particular, if no implementation method is described, output an empty string.

# README
---
dataset_info:
  features:
  - name: content
    dtype: string
  - name: label
    dtype: string
  - name: category
    dtype: string
  - name: dataset
    dtype: string
  - name: node_id
    dtype: int64
  - name: split
    dtype: string
  splits:
  - name: train
    num_bytes: 1006767
    num_examples: 187
  download_size: 505231
  dataset_size: 1006767
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---

Output:
{
    "extracted_code": ""
}
