# Caption Generation from Images

This repository contains the implementation of Capsule Networks on MSCOCO 2017 dataset using tensorflow.


## Dataset Preparation

1. Create a directory named `dataset`.
2. Inside the directory, download and extract the MSCOCO 2017 dataset along with the annotations.
3. Run the file `parse_dataset.py`, it will save the feature vector and the label vector as `X.pickle` and `y.pickle` respectively.
4. Then execute the notebook `capsule_networks.ipynb`
