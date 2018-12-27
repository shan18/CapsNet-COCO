# Caption Generation from Images

This repository contains the implementation of Capsule Networks on MSCOCO 2017 dataset using Keras with Tensorflow Backend.


## Installation

1. Install the COCO API by executing the following commands:
    - `$ git clone https://github.com/cocodataset/cocoapi.git`
    - `$ cd cocoapi/PythonAPI`
    - `$ python setup.py build_ext install`


## Dataset Preparation

1. Go to the directory `dataset`.
2. Inside the directory, download and the MSCOCO 2017 *Train* and *Val* images along with *Train/Val* annotations from [here](http://cocodataset.org/#download) and then extract them.
3. Create training dataset
    - `$ python parse_dataset.py -t train`
4. Create validation dataset
    - `$ python parse_dataset.py -t val`


We used this [repository's](https://github.com/XifengGuo/CapsNet-Keras) implementation of CapsNet.
