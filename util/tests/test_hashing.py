from typing import Any, Callable

import numpy as np
import pytest
import sklearn.utils
from fess38.util.hashing import (
    combine_hashes,
    deterministic_hash,
    deterministic_hash_32,
    deterministic_hash_128,
)


@pytest.mark.parametrize(
    ["value", "hash_fn", "expected"],
    [
        (1, deterministic_hash_32, -68075478),
        (1, sklearn.utils.murmurhash3_32, -68075478),
        (1, deterministic_hash, 19144387141682250),
        (b"", deterministic_hash_128, 0),
        (
            1,
            deterministic_hash_128,
            81803616929829522406399855769982190666,
        ),
    ],
)
def test_deterministic_hash(value: Any, hash_fn: Callable, expected: int | bytes):
    actual = hash_fn(value)
    if isinstance(actual, bytes):
        actual = int.from_bytes(actual, byteorder="little")
    assert actual == expected


@pytest.mark.parametrize(
    ["args", "expected"],
    [
        ([1, 2, 3], 2484014600714012592),
        ([np.array(1), np.array(2), np.array(3)], 2484014600714012592),
        (["a", "b", "c"], 3830352985438859781),
        ([b"d", b"e", b"f"], -6053552617887066361),
        ([b"", b"", b""], 300336696158073414),
        ([None, None, None], 300336696158073414),
    ],
)
def test_combine_hashes(args, expected):
    assert combine_hashes(*args) == expected
