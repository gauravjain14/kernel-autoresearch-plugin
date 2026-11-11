import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
TOY = os.path.join(ROOT, "tasks", "toy")


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a CUDA GPU and takes minutes")
