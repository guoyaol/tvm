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
"""Util functions for benchmarking dynamic shape workloads"""

from typing import Dict, List, Set, Any, Tuple, Union, Optional, NamedTuple

import tvm
from tvm import relax

INPUT_SHAPE_TYPE = List[Tuple[Tuple[int, ...], str]]  # pylint: disable=invalid-name


class DisplayConfig(NamedTuple):
    """Display configuration.

    Parameters
    ----------
    print_out : bool
        Whether to print out the results.
    desc : bool
        Whether to sort results in descending order.
    sort_by : Optional[str]
        Sort results by this key, if None, no sorting.
    hidden_cols : Optional[List[str]]
        Hidden columns, will not be displayed.
    display_cols : Optional[List[str]]
        Display columns only, will not display other columns or hidden columns.
    """

    print_out: bool = False
    desc: bool = True
    sort_by: Optional[str] = None
    hidden_cols: Optional[List[str]] = None
    display_cols: Optional[List[str]] = None


def get_func_name_from_gv(gv: tvm.ir.GlobalVar) -> str:  # pylint: disable=invalid-name
    """Get function name from a global variable.

    Parameters
    ----------
    gv : tvm.ir.GlobalVar
        The given global variable.

    Returns
    -------
    result : str
        The global variable name without the prefix "...@".
    """
    return gv.astext().split("@")[1] if "@" in gv.astext() else gv.astext()


def dym_var_sample_str(sample: Dict[Union[str, tvm.relax.expr.Call], int]) -> str:
    """Convert a variable value sample to a string.

    Parameters
    ----------
    sample : Dict[Union[str, tvm.relax.expr.Call], int]
        Variable value sample, e.g., {n: 64, m: 128} or {"n": 64, "m": 128}

    Returns
    -------
    result : str
        Variable value sample string, e.g., "n=64, m=128"
    """
    return ", ".join([f"{k}={v}" for k, v in sample.items()])


def populuate_input_shape(
    input_infos: List[Union[relax.TensorStructInfo, Tuple[Tuple[Union[int, str], ...], str]]],
    dym_var_sample: Dict[str, int],
) -> INPUT_SHAPE_TYPE:
    """
    Populate input shapes with dynamic shape variable samples.

    Parameters
    ----------
    input_infos : List[Union[relax.TensorStructInfo, Tuple[Tuple[Union[int, str], ...], str]]]
        Input tensor information, including shape and dtype,
        e.g., [..., Shape(1, n, 128) with dtype="int32", ...]
    dym_var_sample : Dict[str, int]
        Dynamic shape variable sample, e.g., {"n": 64}

    Returns
    -------
    results : INPUT_SHAPE_TYPE
        Input shapes with dynamic shape variable samples, e.g.,
        [..., ((1, 64, 128), "int32"), ...] if n=64 or
        [..., (128, "scalar"), ...] if n=128 for scalar input
    """
    results: INPUT_SHAPE_TYPE = []
    for input_info in input_infos:
        shape = []
        if isinstance(input_info, relax.struct_info.ShapeStructInfo):
            # scalar input
            if input_info.values is not None:
                results.append(((dym_var_sample[str(input_info.values[0])],), "scalar"))
            else:
                raise ValueError("Unsupported scalar input", input_info)
        else:
            if isinstance(input_info, relax.TensorStructInfo):
                tensor_shape = input_info.shape
                tensor_dtype = input_info.dtype
                if tensor_shape is None or tensor_dtype is None:
                    raise ValueError("Unsupported tensor input", input_info)
            else:
                tensor_shape, tensor_dtype = input_info  # type: ignore
            for dim in tensor_shape:
                if isinstance(dim, int):
                    shape.append(dim)
                elif isinstance(dim, tvm.tir.IntImm):
                    shape.append(dim.value)
                else:
                    shape.append(dym_var_sample[str(dim)])
            results.append(((*shape,), tensor_dtype))
    return results


def random_dym_var_sample_func(
    dym_vars: Set[str],
    sample_idx: int,  # pylint: disable=unused-argument
    sample_num: int,  # pylint: disable=unused-argument
) -> Dict[str, int]:
    """
    Random dynamic shape variable sample function.
    Sample a random value for each dynamic shape variable.

    Parameters
    ----------
    dym_vars : Set[str]
        Dynamic shape variables, e.g., {"n", "m"}
    sample_idx : int
        Sample index denotes the index the function is called for the same
        dynamic shape variable dictionary & function.

        Here we ignore this argument since we always sample a random value,
        but it can be used to sample different values for different sample
        indices, for example, we can use 2^n for the n-th sample where n is
        the sample index.
    sample_num : int
        Sample number denotes the total number of samples.

        Here we ignore this argument since we always sample a random value,
        but it can be used to sample different values for different sample
        numbers, for example, we can use 2^(n%m) for the n-th sample where n
        is the sample index and m is the sample number.

    Returns
    -------
    result : Dict[str, int]
        Dynamic shape variable sample, e.g., {"n": 64, "m": 128}
    """
    import random  # pylint: disable=import-outside-toplevel

    return {var: random.randint(2, 128) for var in dym_vars}


def display_results(
    bench_results: List[Dict[str, Any]], display_config: Optional[DisplayConfig] = None
):
    """Print benchmark results.

    Parameters
    ----------
    bench_results : List[Dict[str, Any]]
        Benchmark results as dictionary list.
    sort_by : str
        Sort results by this key, if None, no sorting.
    desc : bool
        Whether to sort results in descending order.
    """
    # pylint: disable=invalid-name, import-outside-toplevel
    if display_config is None:
        display_config = DisplayConfig()
    try:
        import pandas as pd

        df = pd.DataFrame()
        for record in bench_results:
            df = pd.concat(
                [df, pd.DataFrame(record, index=[0])],
                ignore_index=True,
            )
        # filter columns
        if display_config.display_cols is not None:
            for col in df.columns:
                if col not in display_config.display_cols:
                    df = df.drop(col, axis=1)
        if display_config.hidden_cols is not None:
            for col in display_config.hidden_cols:
                if col in df.columns:
                    df = df.drop(col, axis=1)
        if display_config.sort_by is not None:
            if display_config.sort_by not in df.columns:
                raise ValueError(f"sort_by key {display_config.sort_by} not in benchmark results")
            df = (
                df.sort_values(display_config.sort_by, ascending=not display_config.desc)
                .reset_index()
                .drop("index", axis=1)
            )
        print(df)
    except ModuleNotFoundError:
        print("Pandas not found, printing results in raw format.")
        keys = []
        if len(bench_results) > 0:
            for key in bench_results[0]:
                keys.append(str(key))
        print("\t".join(keys))
        for record in bench_results:
            values = []
            for key in keys:
                values.append(str(record[key]))
            print("\t".join(values))
    print("\n")
    # pylint: enable=invalid-name, import-outside-toplevel
