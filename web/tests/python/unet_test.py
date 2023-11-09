# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Test relax vm through rpc."""

import tvm
import numpy as np
from tvm import rpc, relax
from tvm.contrib import utils, tvmjs
from tvm.script import relax as R

proxy_host = "127.0.0.1"
proxy_port = 9090

from model import Module

import param_utils



def test_rpc():
    if not tvm.runtime.enabled("rpc"):
        return
    
    mod = Module

    n = 1024
    dtype = "float32"
    temp = utils.tempdir()
    wasm_path = temp.relpath("relax.wasm")
    target = tvm.target.Target("webgpu", host="llvm -mtriple=wasm32-unknown-unknown-wasm")

    print("start building wasm")

    ex = relax.build(mod, target)

    print("build success")

    ex.export_library(wasm_path, fcompile=tvmjs.create_tvmjs_wasm)
    wasm_binary = open(wasm_path, "rb").read()

    remote = rpc.connect(
        proxy_host,
        proxy_port,
        key="wasm",
        session_constructor_args=["rpc.WasmSession", wasm_binary],
    )

    print("remote connected")

    def check(remote):
        dev = remote.webgpu(0)
        # invoke the function
        vm = relax.VirtualMachine(remote.system_lib(), device=dev)

        web_params_dict = param_utils.load_params(artifact_path="dist", device=dev)
        web_unet_param = web_params_dict["unet"]

        #[((1, 4, 64, 64), "float32"), ((), "int32"), ((2, 77, 768), "float32")]
        input1 = np.random.uniform(size=(1, 4, 64, 64)).astype(np.float32)
        input2 = np.random.uniform(size=()).astype(np.int32)
        input3 = np.random.uniform(size=(2, 77, 768)).astype(np.float32)
        web_input1 = tvm.nd.array(input1, dev)
        web_input2 = tvm.nd.array(input2, dev)
        web_input3 = tvm.nd.array(input3, dev)


        print("start web infer")
        vm.set_input("unet", web_input1, web_input2, web_input3, *web_unet_param)
        vm.invoke_stateful("unet")
        web_out = vm.get_outputs("unet")

        print("web infer finish")

        web_numpy = web_out.numpy()

        print("get web out")


        metal_input1 = tvm.nd.array(input1, tvm.metal())
        metal_input2 = tvm.nd.array(input2, tvm.metal())
        metal_input3 = tvm.nd.array(input3, tvm.metal())
        metal_params_dict = param_utils.load_params(artifact_path="dist", device=tvm.metal())

        metal_unet_param = metal_params_dict["unet"]
        metal_mod = Module
        print("start metal build")
        metal_ex = relax.build(metal_mod, target=tvm.target.Target("apple/m1-gpu"))
        print("metal build success")
        metal_vm = relax.VirtualMachine(rt_mod=metal_ex, device=tvm.metal())
        print("start metal infer")
        metal_out = metal_vm["unet"](metal_input1, metal_input2, metal_input3, *metal_unet_param)
        print("metal infer success")

        metal_numpy = metal_out.numpy()

        def mean_absolute_error(y_true, y_pred):
            return np.mean(np.abs(y_true - y_pred))

        # Define the Mean Relative Error (MRE) function
        def mean_relative_error(y_true, y_pred):
            # Avoid division by zero by adding a small constant (epsilon)
            epsilon = np.finfo(float).eps
            return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon)))
        
        print("mean_absolute_error: ", mean_absolute_error(web_numpy, metal_numpy))
        print("mean_relative_error: ", mean_relative_error(web_numpy, metal_numpy))
        np.testing.assert_equal(web_numpy, metal_numpy)
        print("Test pass..")

    check(remote)


test_rpc()