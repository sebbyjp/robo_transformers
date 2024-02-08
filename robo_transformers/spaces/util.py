from gym.spaces import Dict
from typing import Any

def apply_fn(space: Any, fn: callable, prefix: str = ""):
    # print(f'pref '{prefix}  ')
    if isinstance(space, Dict):
        return Dict({k: apply_fn(v, fn, f'{prefix}/{k}') for k, v in space.items()})
    else:
        return prefix