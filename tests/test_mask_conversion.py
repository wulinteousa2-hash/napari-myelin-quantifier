import numpy as np
import pytest

from napari_myelin_quantifier._widget import (
    _as_2d_mask,
    _gray_to_binary_mask,
    _invert_image,
    _to_gray_2d,
)


def test_as_2d_mask_keeps_binary_2d_array():
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    out = _as_2d_mask(mask)
    assert out.shape == (2, 2)
    assert np.array_equal(out, mask)


def test_as_2d_mask_rejects_unsupported_shape():
    with pytest.raises(ValueError, match="Expected a binary 2D mask layer"):
        _as_2d_mask(np.zeros((2, 2, 3), dtype=np.uint8))


def test_to_gray_2d_converts_rgb_mean_channel():
    rgb = np.array(
        [[[0, 0, 0], [30, 60, 90]]],
        dtype=np.uint8,
    )
    out = _to_gray_2d(rgb)
    assert out.shape == (1, 2)
    assert np.allclose(out, np.array([[0.0, 60.0]], dtype=np.float32))


def test_invert_image_preserves_range():
    arr = np.array([[0, 2], [4, 6]], dtype=np.float32)
    out = _invert_image(arr)
    assert np.allclose(out, np.array([[6, 4], [2, 0]], dtype=np.float32))


def test_gray_to_binary_mask_thresholds_simple_signal():
    gray = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [10, 10, 10, 10],
            [10, 10, 10, 10],
        ],
        dtype=np.float32,
    )
    out = _gray_to_binary_mask(gray)
    expected = np.array(
        [
            [False, False, False, False],
            [False, False, False, False],
            [True, True, True, True],
            [True, True, True, True],
        ]
    )
    assert np.array_equal(out, expected)
