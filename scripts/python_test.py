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
        "and.py",
        "cast.py",
        "cos.py",
        "crossentropyloss_backward.py",
        "div.py",
        "equal.py",
        "exp.py",
        "gelu.py",
        "gelu_backward.py",
        "hardswish.py",
        "leaky_relu.py",
        "or.py",
        "relu_backward.py",
        "sigmoid_backward.py",
        "silu.py",
        "sin.py",
        "tanh.py",
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
