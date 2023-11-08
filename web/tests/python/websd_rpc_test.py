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

    def check(remote):
        dev = remote.webgpu(0)
        # invoke the function
        vm = relax.VirtualMachine(remote.system_lib(), device=dev)

        web_params_dict = param_utils.load_params(artifact_path="dist", device=dev)
        web_clip_param = web_params_dict["clip"]

        input_ids = np.random.uniform(size=(1,77)).astype(np.int32)
        web_input_ids = tvm.nd.array(input_ids, dev)


        print("start web infer")
        vm.set_input("main", web_input_ids, web_clip_param)
        vm.invoke_stateful("main")
        web_out = vm.get_outputs("main")

        print("get web out")


        import param_utils
        metal_input_ids = tvm.nd.array(input_ids, tvm.metal())
        metal_params_dict = param_utils.load_params(artifact_path="dist", device=tvm.metal())

        metal_clip_param = metal_params_dict["clip"]
        metal_mod = Module
        print("start metal build")
        metal_ex = relax.build(metal_mod, target=tvm.target.Target("apple/m1-gpu"))
        print("metal build success")
        metal_vm = relax.VirtualMachine(rt_mod=metal_ex, device=tvm.metal())
        print("start metal infer")
        metal_out = metal_vm["clip"](metal_input_ids, metal_clip_param)
        print("metal infer success")



        np.testing.assert_equal(web_out.numpy(), metal_out.numpy())
        print("Test pass..")

    check(remote)


test_rpc()
