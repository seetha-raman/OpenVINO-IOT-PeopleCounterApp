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

    ### TODO: Handle the input stream ###
    if args.input == 'CAM':
        input_stream = 0
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp'):
        input_stream = args.input
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "file doesn't exist"

    ### TODO: Handle the input stream ###
    capture  = cv2.VideoCapture(input_stream)
    capture .open(input_stream)
    w = int(capture .get(3))
    h = int(capture .get(4))

    network_shape = infer_network.get_input_shape()
    in_shape = network_shape['image_tensor']

    request_id = 0

    duration = 0
    prev_duration = 0

    counter = 0
    prev_counter = 0
    report_counter = 0
    total_counter = 0

    ### TODO: Loop until stream is over ###
    while capture .isOpened():
        ### TODO: Read from the video capture ###
        flag, frame = capture .read()
        if not flag:
            break

        ### TODO: Pre-process the image as needed ###
        image = cv2.resize(frame, (in_shape[3], in_shape[2]))
        preproc_image = image.transpose((2, 0, 1))
        preproc_image = preproc_image.reshape(1, *preproc_image.shape)

        ### TODO: Start asynchronous inference for specified request ###
        net_input = {'image_tensor': preproc_image, 'image_info': preproc_image.shape[1:]}
        infer_network.exec_net(request_id, net_input)

        retport_duration = None

        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:

            ### TODO: Get the results of the inference request ###
            network_output = infer_network.get_output()

            ### TODO: Extract any desired stats from the results ###
            pointer = 0
            probs = network_output[0, 0, :, 2]
            for i, p in enumerate(probs):
                if p > prob_threshold:
                    pointer += 1
                    box = network_output[0, 0, i, 3:]
                    p1 = (int(box[0] * w), int(box[1] * h))
                    p2 = (int(box[2] * w), int(box[3] * h))
                    frame = cv2.rectangle(frame, p1, p2, (0, 255, 0), 3)

            if pointer != counter:
                prev_counter = counter
                counter = pointer
                if duration >= 3:
                    prev_duration = duration
                    duration = 0
                else:
                    duration = prev_duration + duration
                    prev_duration = 0
            else:
                duration += 1
                if duration >= 3:
                    report_counter = counter
                    if duration == 3 and counter > prev_counter:
                        total_counter += counter - prev_counter
                    elif duration == 3 and counter < prev_counter:
                        retport_duration = int((prev_duration / 10.0) * 1000)

            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            client.publish('person',
                           payload=json.dumps({
                               'count': report_counter, 'total': total_counter}),
                           qos=0, retain=False)
            if retport_duration is not None:
                client.publish('person/duration',
                               payload=json.dumps({'duration': retport_duration}),
                               qos=0, retain=False)

        ### TODO: Send the frame to the FFMPEG server ###
        #  Resize the frame
        frame = cv2.resize(frame, (768, 432))
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

    capture .release()
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