import numpy as np
import keras.backend as K
from scipy.ndimage.interpolation import map_coordinates

from deform_conv.deform_conv import (
    tf_map_coordinates,
    sp_batch_map_coordinates, tf_batch_map_coordinates,
    sp_batch_map_offsets, tf_batch_map_offsets
)


def test_th_map_coordinates():
    np.random.seed(42)
    input = np.random.random((100, 100))
    coords = (np.random.random((200, 2)) * 99)

    sp_mapped_vals = map_coordinates(input, coords.T, order=1)
    tf_mapped_vals = th_map_coordinates(
        Variable(torch.from_numpy(input)), Variable(torch.from_numpy(coords))
    )
    assert np.allclose(sp_mapped_vals, tf_mapped_vals.data.numpy(), atol=1e-5)


def test_th_batch_map_coordinates():
    np.random.seed(42)
    input = np.random.random((4, 100, 100))
    coords = (np.random.random((4, 200, 2)) * 99)

    sp_mapped_vals = sp_batch_map_coordinates(input, coords)
    tf_mapped_vals = th_batch_map_coordinates(
        Variable(torch.from_numpy(input)), Variable(torch.from_numpy(coords))
    )
    assert np.allclose(sp_mapped_vals, tf_mapped_vals.data.numpy(), atol=1e-5)


def test_th_batch_map_offsets():
    np.random.seed(42)
    input = np.random.random((4, 100, 100))
    offsets = (np.random.random((4, 100, 100, 2)) * 2)

    sp_mapped_vals = sp_batch_map_offsets(input, offsets)
    tf_mapped_vals = th_batch_map_offsets(
        Variable(torch.from_numpy(input)), Variable(torch.from_numpy(offsets))
    )
    assert np.allclose(sp_mapped_vals, tf_mapped_vals.data.numpy(), atol=1e-5)


def test_th_batch_map_offsets_grad():
    np.random.seed(42)
    input = np.random.random((4, 100, 100))
    offsets = (np.random.random((4, 100, 100, 2)) * 2)

    input = Variable(torch.from_numpy(input), requires_grad=True)
    offsets = Variable(torch.from_numpy(offsets), requires_grad=False)

    th_mapped_vals = th_batch_map_offsets(input, offsets)
    # TODO: how to test the gradients?
    i = torch.from_numpy(np.random.random((4, 100, 100)))
    th_mapped_vals.backward(i)
    assert th_mapped_vals.grad is None
