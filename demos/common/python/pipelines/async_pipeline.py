"""
 Copyright (C) 2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import logging
from os import stat_result
import threading
from collections import deque
from typing import Dict, Set

from openvino.inference_engine import InferQueue, StatusCode

### TODO: Remove when fixed by demos
import numpy
###


def parse_devices(device_string):
    colon_position = device_string.find(":")
    if colon_position != -1:
        device_type = device_string[:colon_position]
        if device_type == "HETERO" or device_type == "MULTI":
            comma_separated_devices = device_string[colon_position + 1 :]
            devices = comma_separated_devices.split(",")
            for device in devices:
                parenthesis_position = device.find(":")
                if parenthesis_position != -1:
                    device = device[:parenthesis_position]
            return devices
    return (device_string,)


def parse_value_per_device(devices: Set[str], values_string: str) -> Dict[str, int]:
    """Format: <device1>:<value1>,<device2>:<value2> or just <value>"""
    values_string_upper = values_string.upper()
    result = {}
    device_value_strings = values_string_upper.split(",")
    for device_value_string in device_value_strings:
        device_value_list = device_value_string.split(":")
        if len(device_value_list) == 2:
            if device_value_list[0] in devices:
                result[device_value_list[0]] = int(device_value_list[1])
        elif len(device_value_list) == 1 and device_value_list[0] != "":
            for device in devices:
                result[device] = int(device_value_list[0])
        elif device_value_list[0] != "":
            raise RuntimeError(f"Unknown string format: {values_string}")
    return result


def get_user_config(
    flags_d: str, flags_nstreams: str, flags_nthreads: int
) -> Dict[str, str]:
    config = {}

    devices = set(parse_devices(flags_d))

    device_nstreams = parse_value_per_device(devices, flags_nstreams)
    for device in devices:
        if device == "CPU":  # CPU supports a few special performance-oriented keys
            # limit threading for CPU portion of inference
            if flags_nthreads:
                config["CPU_THREADS_NUM"] = str(flags_nthreads)

            config["CPU_BIND_THREAD"] = "NO"

            # for CPU execution, more throughput-oriented execution via streams
            config["CPU_THROUGHPUT_STREAMS"] = (
                str(device_nstreams[device])
                if device in device_nstreams
                else "CPU_THROUGHPUT_AUTO"
            )
        elif device == "GPU":
            config["GPU_THROUGHPUT_STREAMS"] = (
                str(device_nstreams[device])
                if device in device_nstreams
                else "GPU_THROUGHPUT_AUTO"
            )
            if "MULTI" in flags_d and "CPU" in devices:
                # multi-device execution with the CPU + GPU performs best with GPU throttling hint,
                # which releases another CPU thread (that is otherwise used by the GPU driver for active polling)
                config["CLDNN_PLUGIN_THROTTLE"] = "1"
    return config


class AsyncPipeline:
    def __init__(self, network, model, jobs=1):
        self.model = model
        self.completed_request_results = {}
        self.callback_exceptions = {}

        self.infer_queue = InferQueue(network, jobs)
        self.infer_queue.set_infer_callback(self.inference_completion_callback)

    def inference_completion_callback(self, request, status, userdata):
        try:
            id, meta, preprocessing_meta = userdata
            if status != StatusCode.OK:
                raise RuntimeError(
                    "Infer Request has returned status code {}".format(status)
                )
            raw_outputs = {
                key: blob.buffer for key, blob in request.output_blobs.items()
            }
            self.completed_request_results[id] = (raw_outputs, meta, preprocessing_meta)
        except Exception as e:
            self.callback_exceptions.append(e)

    def async_infer(self, inputs, userdata):
        inputs, preprocessing_meta = self.model.preprocess(inputs)
        ### TODO: Remove when fixed by demos
        inputs["data"] = inputs["data"].astype(numpy.float32)
        ###
        userdata_tuple = tuple(userdata + [preprocessing_meta])
        self.infer_queue.async_infer(inputs=inputs, userdata=userdata_tuple)

    def get_result(self, id):
        if id in self.completed_request_results:
            raw_result, meta, preprocess_meta = self.completed_request_results.pop(id)
            return self.model.postprocess(raw_result, preprocess_meta), meta
        return None

    def is_ready(self):
        return self.infer_queue.is_ready()

    def wait_all(self):
        return self.infer_queue.wait_all()
