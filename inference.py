#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.core = None
        self.network = None

        self.input_blob = None
        self.out_blob = None

        self.ienetwork_exec = None
        self.infer_request_handle = None

    def load_model(self, model, cpu_extension, device):
        ### TODO: Load the model ###
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        self.core = IECore()
        self.network = IENetwork(model=model_xml, weights=model_bin)

        ### TODO: Check for supported layers ###
        ### TODO: Add any necessary extensions ###
        layers_supported = self.core.query_network(self.network, device_name=device)
        layers = self.network.layers.keys()
        if not all(l in layers_supported for l in layers):
            self.core.add_extension(cpu_extension, device)

        # Populate ienetwork, input & output blog for given model
        self.ienetwork_exec = self.core.load_network(self.network, device)
        self.input_blob = next(iter(self.network.inputs))
        self.out_blob = next(iter(self.network.outputs))

        ### TODO: Return the loaded inference plugin ###
        ### Note: You may need to update the function parameters. ###
        return self.core

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        return self.network.inputs[self.input_blob].shape

    def exec_net(self, request_id, image):
        ### TODO: Start an asynchronous request ###
        ### TODO: Return any necessary information ###
        self.infer_request_handle = self.ienetwork_exec.start_async(
            request_id,
            inputs={self.input_blob: image})

        return self.infer_request_handle

    def wait(self):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return self.infer_request_handle.wait()

    def get_output(self):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        return self.infer_request_handle.outputs[self.out_blob]

    def clean(self):
        del self.ienetwork_exec
        del self.network
