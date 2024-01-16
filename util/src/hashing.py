from typing import Callable

import numpy as np
import sklearn.utils
from mmh3 import hash, hash64, hash_bytes

Hashable = int | str | bytes | np.ndarray | None


def _deterministic_hash_impl(
    value: Hashable, seed: int, hash_fn: Callable
) -> int | bytes:
    if value is None:
        value = b""

    if isinstance(value, int):
        if hash_fn in (hash, sklearn.utils):
            value = np.int32(value)
        else:
            value = np.int64(value)

    result = hash_fn(value, seed=seed)
    if isinstance(result, tuple):
        result = result[0]

    return result


def deterministic_hash_32(value: Hashable, seed: int = 0) -> int:
    return _deterministic_hash_impl(value, seed, hash)


def deterministic_hash_64(value: Hashable, seed: int = 0) -> int:
    return _deterministic_hash_impl(value, seed, hash64)


def deterministic_hash_128(value: Hashable, seed: int = 0) -> bytes:
    return _deterministic_hash_impl(value, seed, hash_bytes)


def deterministic_hash(value: Hashable, seed: int = 0) -> int:
    return deterministic_hash_64(value, seed)


def combine_hashes(*args: Hashable, hash_fn=deterministic_hash) -> int | bytes:
    return hash_fn(np.array(list(map(hash_fn, args)), dtype=np.int64))
