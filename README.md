# DADA-AAAI2020
Code release for Discriminative Adversarial Domain Adaptation (AAAI2020).

The paper is avaliable [here](https://arxiv.org/abs/1911.12036).

# Requirements
- Python 3.6.8
- Pytorch 1.0.0

# Dataset
The structure of the dataset should be like
```
VisDA
|_ visda_train
|  |_ aeroplane
|     |_ <im-1-name>.jpg
|     |_ ...
|     |_ <im-N-name>.jpg
|  |_ bicycle
|     |_ <im-1-name>.jpg
|     |_ ...
|     |_ <im-N-name>.jpg
|  |_ ... (omit 9 classes)
|  |_ truck
|     |_ <im-1-name>.jpg
|     |_ ...
|     |_ <im-N-name>.jpg
|_ visda_validation_first6classes
|  |_ aeroplane
|     |_ <im-1-name>.jpg
|     |_ ...
|     |_ <im-N-name>.jpg
|  |_ bicycle
|     |_ <im-1-name>.jpg
|     |_ ...
|     |_ <im-N-name>.jpg
|  |_ ... (omit 3 classes)
|  |_ knife
|     |_ <im-1-name>.jpg
|     |_ ...
|     |_ <im-N-name>.jpg
|_ ...
```
# Training
## [Office-31](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view)
Replace paths and domains in run_office31_partial.sh with those in your own system.
## [VisDA](http://ai.bu.edu/visda-2017/)
Replace paths and domains in run_visda_partial.sh with those in your own system.

# Citation
```
@InProceedings{dada,   
  title={Discriminative Adversarial Domain Adaptation},   
  author={Hui Tang and Kui Jia},   
  booktitle={Association for the Advancement of Artificial Intelligence (AAAI)},   
  year={2020},
}
```
