import numpy as np
import torch
from torch.autograd import Variable
from scipy.ndimage.interpolation import map_coordinates

from torch_deform_conv.deform_conv import (
    th_map_coordinates,
    sp_batch_map_coordinates, th_batch_map_coordinates,
    sp_batch_map_offsets, th_batch_map_offsets
)


def test_th_map_coordinates():
    np.random.seed(42)
    input = np.random.random((100, 100))
    coords = (np.random.random((200, 2)) * 99)

    sp_mapped_vals = map_coordinates(input, coords.T, order=1)
    th_mapped_vals = th_map_coordinates(
        Variable(torch.from_numpy(input)), Variable(torch.from_numpy(coords))
    )
    assert np.allclose(sp_mapped_vals, th_mapped_vals.data.numpy(), atol=1e-5)


def test_th_batch_map_coordinates():
    np.random.seed(42)
    input = np.random.random((4, 100, 100))
    coords = (np.random.random((4, 200, 2)) * 99)

    sp_mapped_vals = sp_batch_map_coordinates(input, coords)
    th_mapped_vals = th_batch_map_coordinates(
        Variable(torch.from_numpy(input)), Variable(torch.from_numpy(coords))
    )
    assert np.allclose(sp_mapped_vals, th_mapped_vals.data.numpy(), atol=1e-5)


def test_th_batch_map_offsets():
    np.random.seed(42)
    input = np.random.random((4, 100, 100))
    offsets = (np.random.random((4, 100, 100, 2)) * 2)

    sp_mapped_vals = sp_batch_map_offsets(input, offsets)
    th_mapped_vals = th_batch_map_offsets(
        Variable(torch.from_numpy(input)), Variable(torch.from_numpy(offsets))
    )
    assert np.allclose(sp_mapped_vals, th_mapped_vals.data.numpy(), atol=1e-5)


def test_th_batch_map_offsets_grad():
    np.random.seed(42)
    input = np.random.random((4, 100, 100))
    offsets = (np.random.random((4, 100, 100, 2)) * 2)

    input = Variable(torch.from_numpy(input), requires_grad=True)
    offsets = Variable(torch.from_numpy(offsets), requires_grad=True)

    th_mapped_vals = th_batch_map_offsets(input, offsets)
    e = torch.from_numpy(np.random.random((4, 100, 100)))
    th_mapped_vals.backward(e)
    assert not np.allclose(input.grad.data.numpy(), 0)
    assert not np.allclose(offsets.grad.data.numpy(), 0)
