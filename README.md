# CapsNet COCO

This repository contains the implementation of Capsule Networks on MSCOCO 2017 dataset using Keras with Tensorflow Backend.

<!-- ## Installation

1. Install the COCO API by executing the following commands:
    - `$ git clone https://github.com/cocodataset/cocoapi.git`
    - `$ cd cocoapi/PythonAPI`
    - `$ python setup.py build_ext install` -->

## Dataset Preparation

1. Go to the directory `dataset`.
2. Inside the directory, download and the MSCOCO 2017 _Train_ and _Val_ images along with _Train/Val_ annotations from [here](http://cocodataset.org/#download) and then extract them.
3. Create a simplified version of MSCOCO 2017 dataset  
   `$ python dataset/parse_coco.py`
4. Preprocess the dataset for training Capsule-Network  
   `$ python capsnet_create_dataset.py`

## Train Capsule-Network

1. To run with the default settings  
   `$ python capsule_network.py`

## Issue

The current architecture of CapsNet is not suitable handling complex real-world images present in the MSCOCO dataset. As a result, the trained model does not converge.

We used this [repository's](https://github.com/XifengGuo/CapsNet-Keras) implementation of CapsNet.
