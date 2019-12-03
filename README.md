# DADA-AAAI2020
Code release for Discriminative Adversarial Domain Adaptation (AAAI2020)

# Requirements
- Python 3.6.8
- Pytorch 1.0.0

# Dataset
The structure of the dataset should be like

VisDA   
|visda_train   
|  |_ aeroplane   
|     |_ \<im-1-na      me\>.jpg   
|     |_ ...   
|     |_ \<im-N-name\>.jpg   
|  |_ bicycle   
|     |_ \<im-1-name\>.jpg   
|     |_ ...   
|     |_ \<im-N-name\>.jpg   
|  ...... (omit 9 classes)   
|  |_ truck   
|     |_ \<im-1-name\>.jpg   
|     |_ ...   
|     |_ \<im-N-name\>.jpg   
|_ visda_validation_first6classes<br />
|  |_ aeroplane<br />
|     |_ \<im-1-name\>.jpg<br />
|     |_ ...<br />
|     |_ \<im-N-name\>.jpg<br />
|  |_ bicycle<br />
|     |_ \<im-1-name\>.jpg<br />
|     |_ ...<br />
|     |_ \<im-N-name\>.jpg<br />
|  ...... (omit 3 classes)<br />
|  |_ knife<br />
|     |_ \<im-1-name\>.jpg<br />
|     |_ ...<br />
|     |_ \<im-N-name\>.jpg <br />
|_ ...   
