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
from torch_deform_conv.utils import transfer_weights

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
for epoch in range(10):
    test(model, test_gen, validation_steps, epoch)
    train(model, train_gen, steps_per_epoch, epoch)


torch.save(model, 'models/cnn.th')

# ---
# Evaluate normal CNN

print('Evaluate normal CNN')
model_cnn = torch.load('models/cnn.th')

test(model_cnn, test_gen, validation_steps, epoch)
# 99.27%
test(model_cnn, test_scaled_gen, validation_steps, epoch)
# 58.83%

# ---
# Deformable CNN

print('Finetune deformable CNN (ConvOffset2D and BatchNorm)')
model = get_deform_cnn(trainable=False)
model = model.cuda()
transfer_weights(model_cnn, model)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(20):
    test(model, test_scaled_gen, validation_steps, epoch)
    train(model, train_scaled_gen, steps_per_epoch, epoch)


torch.save(model, 'models/deform_cnn.th')

# ---
# Evaluate deformable CNN

print('Evaluate deformable CNN')
model = torch.load('models/deform_cnn.th')

test(model, test_gen, validation_steps, epoch)
# xx%
test(model, test_scaled_gen, validation_steps, epoch)
# xx%