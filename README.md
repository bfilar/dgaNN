# Deep Learning for DGA Detection
There has been a ton of research into the application of deep learning models in the security space. An exceedingly common task is classifying Domain Generated Algorithms or DGAs. Since DGAs are small, alphanumeric strings, they are a great toy dataset to test a variety of DL approaches.

This repo seeks to provide code on how to reproduce the models/results mentioned in the following papers:

- [Deep Convolutional Neural Networks for DGA Detection](https://www.stratosphereips.org/publications/2019/5/19/deep-convolutional-neural-networks-for-dga-detection) by the researchers at Stratosphere Lab. 
- [Predicting Domain Generation Algorithms with Long Short-Term Memory Networks](https://arxiv.org/abs/1611.00791)

 __The model(s) are implemented in Pytorch & Tensorflow 2.0__

## Model Architectures

### Convolutional NN
| layer   |     activation      | params |
|----------|:-------------:|------:|
| input |  - | - |
| embedding | - | max_features, 100 |
| conv1d | relu | 256, 4, 1 |
| dense1 | relu | 512 |
| dense2 | sigmoid | 1 |

<br />

### LSTM
| layer   |     activation      | params |
|----------|:-------------:|------:|
| embedding | - | max_f
| lstm | relu | 128 |
| dense1 | sigmoid | 1 |
<br />

## DGA Dataset(s)
There are 11 DGA algorithms in the repo these are taken from the old [Endgame repo](https://github.com/endgameinc/dga_predict).  Some are from the https://github.com/baderj/domain_generation_algorithms
repo. These are called out in each file. This repo usese the same GPL 3.0 license.

## Results

### Binary Classification
| model   |     AUC      |
|----------|:-------------:|
| CNN | 0.97 |
| LSTM | -  |


## References

