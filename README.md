# NekoNet
## Overview
This PyTorch repo contains:
- Mostly code and tools we built to train triplet loss ConvNets that learn vector descriptors for images of cat faces. 
- The cat face dataset we mined and used for training.
- Some models we trained using the framework.

Our work was much inspired by [Adam Klein's report](http://cs230.stanford.edu/projects_fall_2019/reports/26251543.pdf).

Our team's original intent was focused on the cat stuff only, but we believe these tools can be used for training embedding extractor of other objects (e.g., human faces) as long as you have the data.

## Table of contents
1. [Methodology overview](#methodology-overview)
    * [The cat face dataset](#cat-face-dataset)
    * [Model structure and techniques](#model-structure-and-techniques)
2. [Installation](#installation)
3. [Pretrained models](#pretrained-models)
4. [Training your own network](#training-your-own-network)

### Methodology overview
#### Cat face dataset
#### Model structure and techniques
### Installation
### Pretrained models
### Training your own network

## TO-DO:
- Choosing a small backbone model. (done)
- Semi-hard triplet loss. ([alfonmedela's implementation](https://github.com/alfonmedela/triplet-loss-pytorch))
- Dataloader. (done)
- Checkpoint saver. (done)
- Testing utilities. (done)
- Training on sample data. (done)
- Augmentation. (only changes brightness and contrast)
- Tracking of some metrics. (loss only)
- Global orthogonal regularization (done)
- Vector length loss. (skipped)
- Finalizing model structure. (done)
- Training, evaluation and tuning.
