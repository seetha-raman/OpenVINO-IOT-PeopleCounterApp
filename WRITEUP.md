# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## Explaining Custom Layers

OpenVINO supports neural network model layers in multiple frameworks - TensorFlow, Caffe, MXNet, Kaldi and ONYX. The list of known layers is different for each of the supported frameworks
Custom layers are neural network model layers that are not natively supported by a given model framework

Some of the potential reasons for handling custom layers are performing layer operation for specific target hardware. Device specific module implementing custom layers

**How to add custom layers**
 - depends on original model framework
 - Both caffe & Tensorflow - register as extensions
 - cafee-only: use caffee to calcualte output shape
 - TensorFlow-only: Replace subgraph with another

## Comparing Model Performance

I have performed inference with 2 models - *faster_rcnn_inception_v2_coco_2018_01_28* and *ssd_mobilenet_v2_coco_2018_03_29*

*faster_rcnn_inception_v2_coco_2018_01_28* - pre-conversation - 142MB,  post- 50.9 MB
*ssd_mobilenet_v2_coco_2018_03_29* - pre-conversation- 179MB,   post- 64.3 MB

Observed, inference time of both faster_rcnn_inception_v2_coco_2018_01_28 and ssd_mobilenet_v2_coco_2018_03_29 are almost same in CPU mode. 

**faster_rcnn_inception_v2_coco_2018_01_28**

    wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
    tar -xvf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
    python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config faster_rcnn_inception_v2_coco_2018_01_28/pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json --output_dir IR/faster_rcnn_inception_v2_coco_2018_01_28

**ssd_mobilenet_v2_coco_2018_03_29**

    wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
    tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
    /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model=ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config ssd_mobilenet_v2_coco_2018_03_29/pipeline.config --reverse_input_channels --output_dir IR/ssd_mobilenet_v2_coco_2018_03_29

## Assess Model Use Cases

Some of the potential use cases of the people counter app are 
- office - AI based attendance monitoring system
- check-in - automate hotel or party check-in
- manufacturing - monitoring employee performance
- retail - bill counter
- traffic - track the people in signal
- games - monitor number of player or duration at specific spot


## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

- Lighting, focal length effects accuracy. Need to do pre-processing the frame before input to inference engine
- Image size must compatible with model input size. Must resize the image before inference

## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
  
- Model 2: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...

- Model 3: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
