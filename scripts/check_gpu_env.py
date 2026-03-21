import sys

import torch


def main() -> int:
    print(f"python: {sys.version.split()[0]}")
    print(f"torch: {torch.__version__}")
    print(f"torch_cuda: {torch.version.cuda}")
    print(f"cuda_available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("No CUDA device detected by PyTorch.")
        return 1

    device_count = torch.cuda.device_count()
    print(f"cuda_device_count: {device_count}")

    for idx in range(device_count):
        name = torch.cuda.get_device_name(idx)
        capability = torch.cuda.get_device_capability(idx)
        print(f"device[{idx}] name: {name}")
        print(f"device[{idx}] capability: {capability[0]}.{capability[1]}")

    # Simple runtime smoke check on the selected default device.
    x = torch.randn(1024, 1024, device="cuda")
    y = torch.randn(1024, 1024, device="cuda")
    z = x @ y
    print(f"matmul_ok: {tuple(z.shape)} on {z.device}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
