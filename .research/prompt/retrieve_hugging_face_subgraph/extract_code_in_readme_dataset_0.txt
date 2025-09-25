
Input:
From the Hugging Face README provided in “# README,” extract and output only the Python code required for execution. Do not output any other information. In particular, if no implementation method is described, output an empty string.

# README
---
task_categories:
- image-classification
pretty_name: Imagenet 1K webdataset resized to largest side length of 256
size_categories:
- 1M<n<10M
---

This is imagenet1k in webdataset format. Images are stored as jpg files. Every image has been resized to a maximum side length of 256. That means that if an image in the original dataset was 1000 by 500, the new size will be 256 by 128. Images with a maximum side length of under 256 were not resized. 

The total size of all dataset files is 57.8 GB, there are 1,281,167 rows in the training split and 50,000 rows in the validation split.
Output:
{
    "extracted_code": ""
}
