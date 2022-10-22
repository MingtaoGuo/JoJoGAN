# JoJoGAN
Unofficial implementation of the paper: JoJoGAN: One Shot Face Stylization
## Description   
--------------

This repo is mainly to re-implement the face stylization paper based on pretrained stylegan2
- JoJoGAN: [JoJoGAN: One Shot Face Stylization](https://arxiv.org/abs/2112.11641)

|1|2|3|4|
|-|-|-|-|
|![](https://github.com/MingtaoGuo/JoJoGAN/blob/main/IMGS/0.png)|![](https://github.com/MingtaoGuo/JoJoGAN/blob/main/IMGS/1.png)|![](https://github.com/MingtaoGuo/JoJoGAN/blob/main/IMGS/2.png)|![](https://github.com/MingtaoGuo/JoJoGAN/blob/main/IMGS/3.png)|
|![](https://github.com/MingtaoGuo/JoJoGAN/blob/main/IMGS/4.png)|![](https://github.com/MingtaoGuo/JoJoGAN/blob/main/IMGS/5.png)|![](https://github.com/MingtaoGuo/JoJoGAN/blob/main/IMGS/6.png)|![](https://github.com/MingtaoGuo/JoJoGAN/blob/main/IMGS/7.png)|

## Getting Started
### Prerequisites
- Linux or macOS
- NVIDIA GPU + CUDA CuDNN
- Python 3

### Installation
- Clone the repository:
``` 
git clone https://github.com/MingtaoGuo/JoJoGAN.git
cd JoJoGAN
```
- Dependencies:  
We recommend running this repository using [Anaconda](https://docs.anaconda.com/anaconda/install/). 
All dependencies for defining the environment are provided in `environment.yaml`.

### Train
- Download the e4e model from [e4e_ffhq_encode.pt](https://github.com/omertov/encoder4editing)
- Download the stylegan2 model from [stylegan2]() 
- Download the yolo5s-face model from [yolo5s-face]() 
- Put e4e and stylegan2 models into the folder **saved_models**
``` 
python train.py --styles_dir ./styles --num_itr 500  
```
### Inference
``` 
python inference.py --num_sample 10000 
```
## Author 
Mingtao Guo
E-mail: gmt798714378 at hotmail dot com

## Reference
[1]. Chong, Min Jin, and David Forsyth. "Jojogan: One shot face stylization." arXiv preprint arXiv:2112.11641 (2021).
