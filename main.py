"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import time
import socket
import json
import cv2
import os
import sys
import numpy as np
import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                             "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.
    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    model = args.model

    cpu_extension = args.cpu_extension
    device = args.device

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(model, cpu_extension, device)
    n, c, model_h, model_w = infer_network.get_input_shape()

    ### TODO: Handle the input stream ###
    if args.input == 'CAM':
        input_stream = 0
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp'):
        input_stream = args.input
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "file doesn't exist"

    ### TODO: Handle the input stream ###
    capture = cv2.VideoCapture(input_stream)
    capture.open(input_stream)
    source_w = capture.get(3)
    source_h = capture.get(4)

    request_id = 0

    start_time = 0
    last_count = 0
    total_count = 0
    current_count = 0

    # Adjust frame threshold based on model performance
    FRAME_THRESHOLD = 50
    counter = FRAME_THRESHOLD
    frame_count_duration = 0
    ### TODO: Loop until stream is over ###
    while capture.isOpened():
        ### TODO: Read from the video capture ###
        flag, frame = capture.read()
        if not flag:
            break

        frame_count_duration += 1
        ### TODO: Pre-process the image as needed ###
        image = cv2.resize(frame, (model_w, model_h))
        preproc_image = image.transpose((2, 0, 1))
        preproc_image = preproc_image.reshape(n, c, model_h, model_w)

        ### TODO: Start asynchronous inference for specified request ###
        infer_network.exec_net(request_id, preproc_image)

        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:

            ### TODO: Get the results of the inference request ###
            infer_start = time.time()
            network_output = infer_network.get_output()

            ### TODO: Extract any desired stats from the results ###
            out_count = 0
            for out in network_output[0][0]:
                infer_time = time.time() - infer_start
                if out[2] > prob_threshold:
                    xymin = (int(out[3] * source_w), int(out[4] * source_h))
                    xymax = (int(out[5] * source_w), int(out[6] * source_h))
                    cv2.rectangle(frame, xymin, xymax, (0, 255, 0), 2)
                    out_count += 1

            # print inference time in frame
            infer_time_message = "Inference time: {:.3f}ms".format(infer_time * 1000)
            cv2.putText(frame, infer_time_message, (15, 15),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)

            #  check counter and frame threshold for handling fluctuation frames
            if last_count != out_count and counter < 0:
                current_count = out_count
                counter = FRAME_THRESHOLD
            else:
                counter -= 1

            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            # new person enters in video
            if current_count > last_count:
                start_time = time.time()
                total_count += current_count - last_count
                client.publish("person", json.dumps({"total": total_count}))
            # calcualte duration
            elif current_count < last_count:
                # duration = int(time.time() - start_time)
                # total_frame = capture.get(cv2.CAP_PROP_FRAME_COUNT)
                fps = capture.get(cv2.CAP_PROP_FPS)
                duration = frame_count_duration / fps
                client.publish("person/duration",
                               json.dumps({"duration": duration}))
                frame_count_duration = 0

            client.publish("person", json.dumps({"count": current_count}))
            last_count = current_count

        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

    capture.release()
    cv2.destroyAllWindows()
    infer_network.clean()


def main():
    """
    Load the network and parse the output.
    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
