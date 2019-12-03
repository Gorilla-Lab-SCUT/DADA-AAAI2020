# DADA-AAAI2020
Code release for Discriminative Adversarial Domain Adaptation (AAAI2020)

# Requirements
- Python 3.6.8
- Pytorch 1.0.0

# Dataset
The structure of the dataset should be like

VisDA<br />
- visda_train<br />
  - aeroplane   
    - \<im-1-name\>.jpg   
    - ...   
    - \<im-N-name\>.jpg   
  - bicycle   
    - \<im-1-name\>.jpg   
    - ...   
    - \<im-N-name\>.jpg   
  -  ...... (omit 9 classes)   
  - truck   
    - \<im-1-name\>.jpg   
    - ...   
    - \<im-N-name\>.jpg    
- visda_validation_first6classes<br />
  - aeroplane   
    - \<im-1-name\>.jpg   
    - ...   
    - \<im-N-name\>.jpg   
  - bicycle   
    - \<im-1-name\>.jpg   
    - ...   
    - \<im-N-name\>.jpg   
  -  ...... (omit 3 classes)   
  - knife<br />
    - \<im-1-name\>.jpg   
    - ...   
    - \<im-N-name\>.jpg   
- ...<br />
