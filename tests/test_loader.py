import numpy as np
from jaal.data import DataLoader

def test_covers_all_samples():
    x = np.arange(10, dtype=float).reshape(10, 1)
    y = x * 2
    loader = DataLoader(x, y, batch_size=3, shuffle=True, seed=0)
    seen = []
    for xb, yb in loader:
        assert np.allclose(yb, xb * 2)
        seen.extend(xb[:, 0])
    assert sorted(seen) == list(range(10))
    assert len(loader) == 4

def test_shuffle_changes_between_epochs():
    x = np.arange(20, dtype=float).reshape(20, 1)
    loader = DataLoader(x, x, batch_size=5, shuffle=True, seed=0)
    first  = np.concatenate([xb[:, 0] for xb, _ in loader])
    second = np.concatenate([xb[:, 0] for xb, _ in loader])
    assert not np.array_equal(first, second)

def test_no_shuffle_is_repeatable():
    x = np.arange(20, dtype=float).reshape(20, 1)
    loader = DataLoader(x, x, batch_size=5, shuffle=False)
    first  = np.concatenate([xb[:, 0] for xb, _ in loader])
    second = np.concatenate([xb[:, 0] for xb, _ in loader])
    assert np.array_equal(first, second)

def test_inputs_not_modified():
    x = np.arange(10, dtype=float).reshape(10, 1)
    before = x.copy()
    for _ in DataLoader(x, x, batch_size=3, seed=0):
        pass
    assert np.array_equal(x, before)

test_covers_all_samples()
test_shuffle_changes_between_epochs()
test_no_shuffle_is_repeatable()
test_inputs_not_modified()
