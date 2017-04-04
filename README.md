# PyTorch implementation of Deformable Convolution
> by Wei OUYANG

The original implementation in Keras/TensorFlow: https://github.com/felixlaumon/deform-conv


# Understanding Deformable Convolution
> Dai, Jifeng, Haozhi Qi, Yuwen Xiong, Yi Li, Guodong Zhang, Han Hu, and Yichen
Wei. 2017. “Deformable Convolutional Networks.” arXiv [cs.CV]. arXiv.
http://arxiv.org/abs/1703.06211

![](deformable-learned-offset-filtered.gif)

Check out
https://medium.com/@phelixlau/notes-on-deformable-convolutional-networks-baaabbc11cf3
for my summary of the paper.

## Experiment on MNIST and Scaled Data Augmentation

To demonstrate the effectiveness of deformable convolution with scaled images,
we show that by simply replacing regular convolution with deformable convolution
and fine-tuning just the offsets with a scale-augmented datasets, deformable CNN
performs significantly better than regular CNN on the scaled MNIST dataset. This
indicates that deformable convolution is able to more effectively utilize
already learned feature map to represent geometric distortion.

First, we train a 4-layer CNN with regular convolution on MNIST without any data
augmentation. Then we replace all regular convolution layers with deformable
convolution layers and freeze the weights of all layers except the newly added
convolution layers responsible for learning the offsets.  This model is then
fine-tuned on the scale-augmented MNIST dataset.

In this set up, the deformable CNN is forced to make better use of the learned
feature map by only changing its receptive field.

Note that the deformable CNN did not receive additional supervision other than
the labels and is trained with cross-entropy just like the regular CNN.

| Test Accuracy | Regular CNN | Deformable CNN |
| --- | --- | --- |
| Regular MNIST | 98.74% | 97.27% |
| Scaled MNIST | 57.01% | 92.55% |

Please refer to `scripts/scaled_mnist.py` for reproducing this result.

## Notes on Implementation

- This implementation is not efficient. In fact a forward pass with deformed
  convolution takes 260 ms, while regular convolution takes only 10 ms. Also,
  GPU average utilization is only around 10%.
- This implementation also does not take advantage of the fact that offsets and
  the input have similar shape (in `tf_batch_map_offsets`). (So STN-style
  bilinear sampling will help)
- The TensorFlow Keras backend must be used (channel-last)
- You can check ensure the TensorFlow implementation is equivalent to its scipy
  counterpart by running unit tests (e.g. `py.test -x -v --pdb`)
