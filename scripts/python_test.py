import os
import subprocess
from set_env import set_env
import sys

PROJECT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "test", "infiniop")
)
os.chdir(PROJECT_DIR)


def run_tests(args):
    failed = []
    for test in [
        "add.py",
        "and.py",
        "attention.py",
        "batch_norm.py",
        "cast.py",
        "causal_softmax.py",
        "clip.py",
        "cos.py",
        "crossentropyloss_backward.py",
        "div.py",
        "equal.py",
        "exp.py",
        "gather.py",
        "gelu.py",
        "gelu_backward.py",
        "gemm.py",
        "hardswish.py",
        "index_copy_inplace.py",
        "layer_norm.py",
        "leaky_relu.py",
        "linear.py",
        "linear_backward.py",
        "mul.py",
        "or.py",
        "random_sample.py",
        "rearrange.py",
        "reduce_max.py",
        "reduce_mean.py",
        "relu_backward.py",
        "rms_norm.py",
        "rms_norm_backward.py",
        "rope.py",
        "scatter.py",
        "sigmoid_backward.py",
        "silu.py",
        "sin.py",
        "sub.py",
        "swiglu.py",
        "tanh.py",
        "tril.py",
        "triu.py",
        "where.py",
    ]:
        result = subprocess.run(
            f"python {test} {args} --debug", text=True, encoding="utf-8", shell=True
        )
        if result.returncode != 0:
            failed.append(test)

    return failed


if __name__ == "__main__":
    set_env()
    failed = run_tests(" ".join(sys.argv[1:]))
    if len(failed) == 0:
        print("\033[92mAll tests passed!\033[0m")
    else:
        print("\033[91mThe following tests failed:\033[0m")
        for test in failed:
            print(f"\033[91m - {test}\033[0m")
    exit(len(failed))
