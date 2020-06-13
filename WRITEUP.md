
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
 - cafee-only: use caffee to calculate output shape
 - TensorFlow-only: Replace subgraph with another

## Comparing Model Performance

I have performed inference with 3 models - *ssd_mobilenet_v2_coco_2018_03_29*, *ssdlite_mobilenet_v2_coco_2018_05_09* and *faster_rcnn_inception_v2_coco_2018_01_28*

**COCO Models**
*ssd_mobilenet_v2_coco_2018_03_29* - pre-conversation- *179 MB*,   post- *64.3 MB* , *380-470 ms* 
*ssdlite_mobilenet_v2_coco_2018_05_09* - pre-conversation- *60 MB*,   post- *18 MB* , *610-710 ms*
*faster_rcnn_inception_v2_coco_2018_01_28* - pre-conversation - *142 MB*,  post- *50.9 MB* , *700-800 ms* 

**Intel pre-trained Models** 
*person-detection-retail-0013* - size *50.9 MB* , *610-730 ms* 


## Assess Model Use Cases

Some of the potential use cases of the people counter app are 
- office - AI based attendance monitoring system
- check-in - automate hotel or party check-in
- manufacturing - monitoring employee performance
- retail - bill counter
- traffic - track the people in signal
- games - monitor number of player or duration at specific spot


## Assess Effects on End User Needs

Yes. Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

- lighting - introduced deep shadow, leads skew result
- model accuracy - as seen, accuracy effects total count and average time due to fluctuation of detection
- camera focal length - leads blurring image 
- image size - too low image size leads low resolution 
- few others - Noise, angles

Must handle these scenarios in input before feed into inference by pre-processing like rank filtering, color correction, sharpen filter, resize image etc. 

## Model Research

Some times, model misses the detection of the people on few frames. Few ways to address this problem:
 - choose high accuracy model 
 - higher hardware 
 - train the model with respective data set 
 - setting frame threshold to handle fluctuation frames 
 - work with probability threshold for detection filtering 
 - implement advanced strategy like derive count by tracking coordinate of people in and out 

In investigating potential people counter models, I tried each of the following three models:

**person-detection-retail-0013**

    /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name person-detection-retail-0013 -o models/IR/pd-retail-0013 
    python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m models/IR/pd-retail-0013/intel/person-detection-retail-0013/FP16/person-detection-retail-0013.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
    
**ssd_mobilenet_v2_coco_2018_03_29**

    wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
    tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
    /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model=ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config ssd_mobilenet_v2_coco_2018_03_29/pipeline.config --reverse_input_channels --output_dir IR/ssd_mobilenet_v2_coco_2018_03_29  
    python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m models/IR/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm


**ssdlite_mobilenet_v2_coco_2018_05_09**

    wget http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz  
    tar -xvf ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz  
    /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model=models/ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config models/ssdlite_mobilenet_v2_coco_2018_05_09/pipeline.config --reverse_input_channels --output_dir models/IR/ssdlite_mobilenet_v2_coco_2018_05_09  
    python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m models/IR/ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

**faster_rcnn_inception_v2_coco_2018_01_28**

    wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
    tar -xvf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
    python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config faster_rcnn_inception_v2_coco_2018_01_28/pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json --output_dir IR/faster_rcnn_inception_v2_coco_2018_01_28  
    python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m models/IR/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm