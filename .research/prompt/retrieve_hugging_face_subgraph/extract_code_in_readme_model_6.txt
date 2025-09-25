
Input:
From the Hugging Face README provided in “# README,” extract and output only the Python code required for execution. Do not output any other information. In particular, if no implementation method is described, output an empty string.

# README
---
library_name: pytorch
license: other
tags:
- backbone
- real_time
- android
pipeline_tag: image-classification

---

![](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/mobilenet_v3_small/web-assets/model_demo.png)

# MobileNet-v3-Small: Optimized for Mobile Deployment
## Imagenet classifier and general purpose backbone


MobileNetV3Small is a machine learning model that can classify images from the Imagenet dataset. It can also be used as a backbone in building more complex models for specific use cases.

This model is an implementation of MobileNet-v3-Small found [here](https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py).


This repository provides scripts to run MobileNet-v3-Small on Qualcomm® devices.
More details on model performance across various devices, can be found
[here](https://aihub.qualcomm.com/models/mobilenet_v3_small).



### Model Details

- **Model Type:** Model_use_case.image_classification
- **Model Stats:**
  - Model checkpoint: Imagenet
  - Input resolution: 224x224
  - Number of parameters: 2.54M
  - Model size (float): 9.71 MB

| Model | Precision | Device | Chipset | Target Runtime | Inference Time (ms) | Peak Memory Range (MB) | Primary Compute Unit | Target Model
|---|---|---|---|---|---|---|---|---|
| MobileNet-v3-Small | float | QCS8275 (Proxy) | Qualcomm® QCS8275 (Proxy) | TFLITE | 2.038 ms | 0 - 21 MB | NPU | [MobileNet-v3-Small.tflite](https://huggingface.co/qualcomm/MobileNet-v3-Small/blob/main/MobileNet-v3-Small.tflite) |
| MobileNet-v3-Small | float | QCS8275 (Proxy) | Qualcomm® QCS8275 (Proxy) | QNN_DLC | 1.965 ms | 1 - 22 MB | NPU | [MobileNet-v3-Small.dlc](https://huggingface.co/qualcomm/MobileNet-v3-Small/blob/main/MobileNet-v3-Small.dlc) |
| MobileNet-v3-Small | float | QCS8450 (Proxy) | Qualcomm® QCS8450 (Proxy) | TFLITE | 0.985 ms | 0 - 34 MB | NPU | [MobileNet-v3-Small.tflite](https://huggingface.co/qualcomm/MobileNet-v3-Small/blob/main/MobileNet-v3-Small.tflite) |
| MobileNet-v3-Small | float | QCS8450 (Proxy) | Qualcomm® QCS8450 (Proxy) | QNN_DLC | 1.406 ms | 1 - 35 MB | NPU | [MobileNet-v3-Small.dlc](https://huggingface.co/qualcomm/MobileNet-v3-Small/blob/main/MobileNet-v3-Small.dlc) |
| MobileNet-v3-Small | float | QCS8550 (Proxy) | Qualcomm® QCS8550 (Proxy) | TFLITE | 0.776 ms | 0 - 50 MB | NPU | [MobileNet-v3-Small.tflite](https://huggingface.co/qualcomm/MobileNet-v3-Small/blob/main/MobileNet-v3-Small.tflite) |
| MobileNet-v3-Small | float | QCS8550 (Proxy) | Qualcomm® QCS8550 (Proxy) | QNN_DLC | 0.771 ms | 0 - 48 MB | NPU | [MobileNet-v3-Small.dlc](https://huggingface.co/qualcomm/MobileNet-v3-Small/blob/main/MobileNet-v3-Small.dlc) |
| MobileNet-v3-Small | float | QCS8550 (Proxy) | Qualcomm® QCS8550 (Proxy) | ONNX | 0.612 ms | 0 - 53 MB | NPU | [MobileNet-v3-Small.onnx.zip](https://huggingface.co/qualcomm/MobileNet-v3-Small/blob/main/MobileNet-v3-Small.onnx.zip) |
| MobileNet-v3-Small | float | QCS9075 (Proxy) | Qualcomm® QCS9075 (Proxy) | TFLITE | 1.085 ms | 0 - 21 MB | NPU | [MobileNet-v3-Small.tflite](https://huggingface.co/qualcomm/MobileNet-v3-Small/blob/main/MobileNet-v3-Small.tflite) |
| MobileNet-v3-Small | float | QCS9075 (Proxy) | Qualcomm® QCS9075 (Proxy) | QNN_DLC | 1.071 ms | 1 - 22 MB | NPU | [MobileNet-v3-Small.dlc](https://huggingface.co/qualcomm/MobileNet-v3-Small/blob/main/MobileNet-v3-Small.dlc) |
| MobileNet-v3-Small | float | SA7255P ADP | Qualcomm® SA7255P | TFLITE | 2.038 ms | 0 - 21 MB | NPU | [MobileNet-v3-Small.tflite](https://huggingface.co/qualcomm/MobileNet-v3-Small/blob/main/MobileNet-v3-Small.tflite) |
| MobileNet-v3-Small | float | SA7255P ADP | Qualcomm® SA7255P | QNN_DLC | 1.965 ms | 1 - 22 MB | NPU | [MobileNet-v3-Small.dlc](https://huggingface.co/qualcomm/MobileNet-v3-Small/blob/main/MobileNet-v3-Small.dlc) |
| MobileNet-v3-Small | float | SA8255 (Proxy) | Qualcomm® SA8255P (Proxy) | TFLITE | 0.779 ms | 0 - 50 MB | NPU | [MobileNet-v3-Small.tflite](https://huggingface.co/qualcomm/MobileNet-v3-Small/blob/main/MobileNet-v3-Small.tflite) |
| MobileNet-v3-Small | float | SA8255 (Proxy) | Qualcomm® SA8255P (Proxy) | QNN_DLC | 0.772 ms | 1 - 7 MB | NPU | [MobileNet-v3-Small.dlc](https://huggingface.co/qualcomm/MobileNet-v3-Small/blob/main/MobileNet-v3-Small.dlc) |
| MobileNet-v3-Small | float | SA8295P ADP | Qualcomm® SA8295P | TFLITE | 1.4 ms | 0 - 28 MB | NPU | [MobileNet-v3-Small.tflite](https://huggingface.co/qualcomm/MobileNet-v3-Small/blob/main/MobileNet-v3-Small.tflite) |
| MobileNet-v3-Small | float | SA8295P ADP | Qualcomm® SA8295P | QNN_DLC | 1.365 ms | 0 - 29 MB | NPU | [MobileNet-v3-Small.dlc](https://huggingface.co/qualcomm/MobileNet-v3-Small/blob/main/MobileNet-v3-Small.dlc) |
| MobileNet-v3-Small | float | SA8650 (Proxy) | Qualcomm® SA8650P (Proxy) | TFLITE | 0.778 ms | 0 - 50 MB | NPU | [MobileNet-v3-Small.tflite](https://huggingface.co/qualcomm/MobileNet-v3-Small/blob/main/MobileNet-v3-Small.tflite) |
| MobileNet-v3-Small | float | SA8650 (Proxy) | Qualcomm® SA8650P (Proxy) | QNN_DLC | 0.773 ms | 0 - 46 MB | NPU | [MobileNet-v3-Small.dlc](https://huggingface.co/qualcomm/MobileNet-v3-Small/blob/main/MobileNet-v3-Small.dlc) |
| MobileNet-v3-Small | float | SA8775P ADP | Qualcomm® SA8775P | TFLITE | 1.085 ms | 0 - 21 MB | NPU | [MobileNet-v3-Small.tflite](https://huggingface.co/qualcomm/MobileNet-v3-Small/blob/main/MobileNet-v3-Small.tflite) |
| MobileNet-v3-Small | float | SA8775P ADP | Qualcomm® SA8775P | QNN_DLC | 1.071 ms | 1 - 22 MB | NPU | [MobileNet-v3-Small.dlc](https://huggingface.co/qualcomm/MobileNet-v3-Small/blob/main/MobileNet-v3-Small.dlc) |
| MobileNet-v3-Small | float | Samsung Galaxy S23 | Snapdragon® 8 Gen 2 Mobile | TFLITE | 0.777 ms | 0 - 50 MB | NPU | [MobileNet-v3-Small.tflite](https://huggingface.co/qualcomm/MobileNet-v3-Small/blob/main/MobileNet-v3-Small.tflite) |
| MobileNet-v3-Small | float | Samsung Galaxy S23 | Snapdragon® 8 Gen 2 Mobile | QNN_DLC | 0.772 ms | 0 - 47 MB | NPU | [MobileNet-v3-Small.dlc](https://huggingface.co/qualcomm/MobileNet-v3-Small/blob/main/MobileNet-v3-Small.dlc) |
| MobileNet-v3-Small | float | Samsung Galaxy S23 | Snapdragon® 8 Gen 2 Mobile | ONNX | 0.628 ms | 0 - 39 MB | NPU | [MobileNet-v3-Small.onnx.zip](https://huggingface.co/qualcomm/MobileNet-v3-Small/blob/main/MobileNet-v3-Small.onnx.zip) |
| MobileNet-v3-Small | float | Samsung Galaxy S24 | Snapdragon® 8 Gen 3 Mobile | TFLITE | 0.511 ms | 0 - 35 MB | NPU | [MobileNet-v3-Small.tflite](https://huggingface.co/qualcomm/MobileNet-v3-Small/blob/main/MobileNet-v3-Small.tflite) |
| MobileNet-v3-Small | float | Samsung Galaxy S24 | Snapdragon® 8 Gen 3 Mobile | QNN_DLC | 0.508 ms | 1 - 33 MB | NPU | [MobileNet-v3-Small.dlc](https://huggingface.co/qualcomm/MobileNet-v3-Small/blob/main/MobileNet-v3-Small.dlc) |
| MobileNet-v3-Small | float | Samsung Galaxy S24 | Snapdragon® 8 Gen 3 Mobile | ONNX | 0.434 ms | 0 - 37 MB | NPU | [MobileNet-v3-Small.onnx.zip](https://huggingface.co/qualcomm/MobileNet-v3-Small/blob/main/MobileNet-v3-Small.onnx.zip) |
| MobileNet-v3-Small | float | Snapdragon 8 Elite QRD | Snapdragon® 8 Elite Mobile | TFLITE | 0.497 ms | 0 - 24 MB | NPU | [MobileNet-v3-Small.tflite](https://huggingface.co/qualcomm/MobileNet-v3-Small/blob/main/MobileNet-v3-Small.tflite) |
| MobileNet-v3-Small | float | Snapdragon 8 Elite QRD | Snapdragon® 8 Elite Mobile | QNN_DLC | 0.476 ms | 0 - 24 MB | NPU | [MobileNet-v3-Small.dlc](https://huggingface.co/qualcomm/MobileNet-v3-Small/blob/main/MobileNet-v3-Small.dlc) |
| MobileNet-v3-Small | float | Snapdragon 8 Elite QRD | Snapdragon® 8 Elite Mobile | ONNX | 0.373 ms | 0 - 24 MB | NPU | [MobileNet-v3-Small.onnx.zip](https://huggingface.co/qualcomm/MobileNet-v3-Small/blob/main/MobileNet-v3-Small.onnx.zip) |
| MobileNet-v3-Small | float | Snapdragon X Elite CRD | Snapdragon® X Elite | QNN_DLC | 0.916 ms | 47 - 47 MB | NPU | [MobileNet-v3-Small.dlc](https://huggingface.co/qualcomm/MobileNet-v3-Small/blob/main/MobileNet-v3-Small.dlc) |
| MobileNet-v3-Small | float | Snapdragon X Elite CRD | Snapdragon® X Elite | ONNX | 0.657 ms | 5 - 5 MB | NPU | [MobileNet-v3-Small.onnx.zip](https://huggingface.co/qualcomm/MobileNet-v3-Small/blob/main/MobileNet-v3-Small.onnx.zip) |




## Installation


Install the package via pip:
```bash
pip install qai-hub-models
```


## Configure Qualcomm® AI Hub to run this model on a cloud-hosted device

Sign-in to [Qualcomm® AI Hub](https://app.aihub.qualcomm.com/) with your
Qualcomm® ID. Once signed in navigate to `Account -> Settings -> API Token`.

With this API token, you can configure your client to run models on the cloud
hosted devices.
```bash
qai-hub configure --api_token API_TOKEN
```
Navigate to [docs](https://app.aihub.qualcomm.com/docs/) for more information.



## Demo off target

The package contains a simple end-to-end demo that downloads pre-trained
weights and runs this model on a sample input.

```bash
python -m qai_hub_models.models.mobilenet_v3_small.demo
```

The above demo runs a reference implementation of pre-processing, model
inference, and post processing.

**NOTE**: If you want running in a Jupyter Notebook or Google Colab like
environment, please add the following to your cell (instead of the above).
```
%run -m qai_hub_models.models.mobilenet_v3_small.demo
```


### Run model on a cloud-hosted device

In addition to the demo, you can also run the model on a cloud-hosted Qualcomm®
device. This script does the following:
* Performance check on-device on a cloud-hosted device
* Downloads compiled assets that can be deployed on-device for Android.
* Accuracy check between PyTorch and on-device outputs.

```bash
python -m qai_hub_models.models.mobilenet_v3_small.export
```



## How does this work?

This [export script](https://aihub.qualcomm.com/models/mobilenet_v3_small/qai_hub_models/models/MobileNet-v3-Small/export.py)
leverages [Qualcomm® AI Hub](https://aihub.qualcomm.com/) to optimize, validate, and deploy this model
on-device. Lets go through each step below in detail:

Step 1: **Compile model for on-device deployment**

To compile a PyTorch model for on-device deployment, we first trace the model
in memory using the `jit.trace` and then call the `submit_compile_job` API.

```python
import torch

import qai_hub as hub
from qai_hub_models.models.mobilenet_v3_small import Model

# Load the model
torch_model = Model.from_pretrained()

# Device
device = hub.Device("Samsung Galaxy S24")

# Trace model
input_shape = torch_model.get_input_spec()
sample_inputs = torch_model.sample_inputs()

pt_model = torch.jit.trace(torch_model, [torch.tensor(data[0]) for _, data in sample_inputs.items()])

# Compile model on a specific device
compile_job = hub.submit_compile_job(
    model=pt_model,
    device=device,
    input_specs=torch_model.get_input_spec(),
)

# Get target model to run on-device
target_model = compile_job.get_target_model()

```


Step 2: **Performance profiling on cloud-hosted device**

After compiling models from step 1. Models can be profiled model on-device using the
`target_model`. Note that this scripts runs the model on a device automatically
provisioned in the cloud.  Once the job is submitted, you can navigate to a
provided job URL to view a variety of on-device performance metrics.
```python
profile_job = hub.submit_profile_job(
    model=target_model,
    device=device,
)
        
```

Step 3: **Verify on-device accuracy**

To verify the accuracy of the model on-device, you can run on-device inference
on sample input data on the same cloud hosted device.
```python
input_data = torch_model.sample_inputs()
inference_job = hub.submit_inference_job(
    model=target_model,
    device=device,
    inputs=input_data,
)
    on_device_output = inference_job.download_output_data()

```
With the output of the model, you can compute like PSNR, relative errors or
spot check the output with expected output.

**Note**: This on-device profiling and inference requires access to Qualcomm®
AI Hub. [Sign up for access](https://myaccount.qualcomm.com/signup).



## Run demo on a cloud-hosted device

You can also run the demo on-device.

```bash
python -m qai_hub_models.models.mobilenet_v3_small.demo --eval-mode on-device
```

**NOTE**: If you want running in a Jupyter Notebook or Google Colab like
environment, please add the following to your cell (instead of the above).
```
%run -m qai_hub_models.models.mobilenet_v3_small.demo -- --eval-mode on-device
```


## Deploying compiled model to Android


The models can be deployed using multiple runtimes:
- TensorFlow Lite (`.tflite` export): [This
  tutorial](https://www.tensorflow.org/lite/android/quickstart) provides a
  guide to deploy the .tflite model in an Android application.


- QNN (`.so` export ): This [sample
  app](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/sample_app.html)
provides instructions on how to use the `.so` shared library  in an Android application.


## View on Qualcomm® AI Hub
Get more details on MobileNet-v3-Small's performance across various devices [here](https://aihub.qualcomm.com/models/mobilenet_v3_small).
Explore all available models on [Qualcomm® AI Hub](https://aihub.qualcomm.com/)


## License
* The license for the original implementation of MobileNet-v3-Small can be found
  [here](https://github.com/pytorch/vision/blob/main/LICENSE).
* The license for the compiled assets for on-device deployment can be found [here](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/Qualcomm+AI+Hub+Proprietary+License.pdf)



## References
* [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)
* [Source Model Implementation](https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py)



## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).



Output:
{
    "extracted_code": "import torch\n\nimport qai_hub as hub\nfrom qai_hub_models.models.mobilenet_v3_small import Model\n\n# Load the model\ntorch_model = Model.from_pretrained()\n\n# Device\ndevice = hub.Device(\"Samsung Galaxy S24\")\n\n# Trace model\ninput_shape = torch_model.get_input_spec()\nsample_inputs = torch_model.sample_inputs()\n\npt_model = torch.jit.trace(torch_model, [torch.tensor(data[0]) for _, data in sample_inputs.items()])\n\n# Compile model on a specific device\ncompile_job = hub.submit_compile_job(\n    model=pt_model,\n    device=device,\n    input_specs=torch_model.get_input_spec(),\n)\n\n# Get target model to run on-device\ntarget_model = compile_job.get_target_model()\n\nprofile_job = hub.submit_profile_job(\n    model=target_model,\n    device=device,\n)\n\ninput_data = torch_model.sample_inputs()\ninference_job = hub.submit_inference_job(\n    model=target_model,\n    device=device,\n    inputs=input_data,\n)\n    on_device_output = inference_job.download_output_data()\n"
}
