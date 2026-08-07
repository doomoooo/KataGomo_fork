import argparse

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


@triton.jit
def add_kernel(a, b, c):
    index = tl.arange(0, 32)
    tl.store(c + index, tl.load(a + index) + tl.load(b + index))


parser = argparse.ArgumentParser()
parser.add_argument("--arch", required=True, type=int)
args = parser.parse_args()

compiled = triton.compile(
    ASTSource(
        fn=add_kernel,
        signature={"a": "*fp32", "b": "*fp32", "c": "*fp32"},
        constexprs={},
    ),
    target=GPUTarget("cuda", args.arch, 32),
)
assert compiled.asm["cubin"]
print("TRITON_COMPILE_OK", args.arch, len(compiled.asm["cubin"]))
