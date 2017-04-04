from __future__ import absolute_import, division
# %env CUDA_VISIBLE_DEVICES=0

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from torch_deform_conv.layers import ConvOffset2D
from torch_deform_conv.cnn import get_cnn, get_deform_cnn
from torch_deform_conv.mnist import get_gen


batch_size = 32
n_train = 60000
n_test = 10000
steps_per_epoch = int(np.ceil(n_train / batch_size))
validation_steps = int(np.ceil(n_test / batch_size))

train_gen = get_gen(
    'train', batch_size=batch_size,
    scale=(1.0, 1.0), translate=0.0,
    shuffle=True
)
test_gen = get_gen(
    'test', batch_size=batch_size,
    scale=(1.0, 1.0), translate=0.0,
    shuffle=False
)
train_scaled_gen = get_gen(
    'train', batch_size=batch_size,
    scale=(1.0, 2.5), translate=0.2,
    shuffle=True
)
test_scaled_gen = get_gen(
    'test', batch_size=batch_size,
    scale=(1.0, 2.5), translate=0.2,
    shuffle=False
)


def train(model, generator, batch_num, epoch):
    model.train()
    for batch_idx in range(batch_num):
        data, target = next(generator)
        data, target = torch.from_numpy(data), torch.from_numpy(target)
        # convert BHWC to BCHW
        data = data.permute(0, 3, 1, 2)
        data, target = data.float().cuda(), target.long().cuda()

        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

    print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, loss.data[0]))

def test(model, generator, batch_num, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for batch_idx in range(batch_num):
        data, target = next(generator)
        data, target = torch.from_numpy(data), torch.from_numpy(target)
        # convert BHWC to BCHW
        data = data.permute(0, 3, 1, 2)
        data, target = data.float().cuda(), target.long().cuda()

        data, target = Variable(data), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /=  batch_num# loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, n_test, 100. * correct / n_test))


# ---
# Normal CNN


model = get_cnn()
model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# optim = SGD(1e-3, momentum=0.99, nesterov=True)
for epoch in range(20):
    test(model, test_gen, validation_steps, epoch)
    train(model, train_gen, steps_per_epoch, epoch)


torch.save(model, 'models/cnn.th')
# 1875/1875 [==============================] - 24s - loss: 0.0090 - acc: 0.9969 - val_loss: 0.0528 - val_acc: 0.9858

# ---
# Evaluate normal CNN

print('Evaluate normal CNN')
model = torch.load('models/cnn.th')

test(model, test_gen, validation_steps, epoch)
# 99.27%
test(model, test_scaled_gen, validation_steps, epoch)
# 58.83%

# ---
# Deformable CNN

model = get_deform_cnn()
model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# optim = SGD(1e-3, momentum=0.99, nesterov=True)
for epoch in range(20):
    test(model, test_gen, validation_steps, epoch)
    train(model, train_gen, steps_per_epoch, epoch)


torch.save(model, 'models/deform_cnn.th')
# 1875/1875 [==============================] - 24s - loss: 0.0090 - acc: 0.9969 - val_loss: 0.0528 - val_acc: 0.9858

# ---
# Evaluate deformable CNN

print('Evaluate deformable CNN')
model = torch.load('models/deform_cnn.th')

test(model, test_gen, validation_steps, epoch)
# 99.11%
test(model, test_scaled_gen, validation_steps, epoch)
# 63.27%

# TODO: support fine-tuning
# deform_conv_layers = [l for l in model.layers if isinstance(l, ConvOffset2D)]
#
# Xb, Yb = next(test_gen)
# for l in deform_conv_layers:
#     print(l)
#     _model = Model(inputs=inputs, outputs=l.output)
#     offsets = _model.predict(Xb)
#     offsets = offsets.reshape(offsets.shape[0], offsets.shape[1], offsets.shape[2], -1, 2)
#     print(offsets.min())
#     print(offsets.mean())
#     print(offsets.max())
