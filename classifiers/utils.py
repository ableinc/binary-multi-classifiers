import torch

# Setup multi-GPU if available
def setup_device():
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        device = torch.device('cuda')
        return device, True
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device, False

__all__ = ["setup_device"]
