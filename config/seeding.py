import os
import random

import numpy as np
import torch

SEED = int(os.getenv("SEED", 42))


def fix_seed(seed=SEED):
    """Fix random seed for reproducibility across all libraries"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
