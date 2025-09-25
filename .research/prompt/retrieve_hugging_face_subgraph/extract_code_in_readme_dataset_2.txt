
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
    num_bytes: 2063874
    num_examples: 265
  download_size: 968161
  dataset_size: 2063874
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
