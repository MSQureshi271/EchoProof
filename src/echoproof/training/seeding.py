"""Seeds for independent fixture, crop, augmentation and loader streams."""
import hashlib
import random


def named_seed(seed: int, name: str) -> int:
    if type(seed) is not int or seed < 0 or not isinstance(name, str) or not name:
        raise ValueError("seed must be nonnegative and name must be nonempty")
    digest = hashlib.sha256(f"echoproof-seed-v1:{seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def seed_everything(seed: int):
    import numpy as np
    import torch

    base = named_seed(seed, "global")
    random.seed(base)
    np.random.seed(base % (2**32))
    torch.manual_seed(base)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return {name: named_seed(seed, name) for name in ("global", "fixtures", "crops", "augmentation", "loader")}


def loader_generator(seed: int):
    import torch
    return torch.Generator().manual_seed(named_seed(seed, "loader"))


def seed_worker(worker_id: int):
    """DataLoader callback; torch assigns each worker's seed before invocation."""
    import numpy as np
    import torch
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
