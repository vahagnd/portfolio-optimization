import torch


def get_device():
    """Get appropriate device (mps for Mac, cuda for NVIDIA, cpu fallback)"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()
