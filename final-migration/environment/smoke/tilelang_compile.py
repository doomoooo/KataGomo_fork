import argparse

import tilelang
import tilelang.language as T
from tvm.target import Target


@T.prim_func
def add_kernel(
    a: T.Tensor((128,), "float32"),
    b: T.Tensor((128,), "float32"),
    c: T.Tensor((128,), "float32"),
):
    with T.Kernel(1, threads=128):
        for i in T.Parallel(128):
            c[i] = a[i] + b[i]


parser = argparse.ArgumentParser()
parser.add_argument("--arch", required=True)
args = parser.parse_args()

kernel = tilelang.compile(
    add_kernel,
    out_idx=[2],
    target=Target({"kind": "cuda", "arch": f"sm_{args.arch}"}),
)
print("TILELANG_COMPILE_OK", args.arch, type(kernel).__name__)
