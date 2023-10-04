# SelfHDR

PyTorch implementation of [**Self-Supervised High Dynamic Range Imaging with Multi-Exposure Images in Dynamic Scenes**](https://arxiv.org/abs/2310.01840) 

- In this work, we take advantage of the internal characteristics of the multi-exposure images to propose a self-supervised HDR reconstruction approach SelfHDR, which achieves comparable performance to supervised ones.

- Codes will be released in a few weeks.


## 1. Abstract

Merging multi-exposure images is a common approach for obtaining high dynamic range (HDR) images, with the primary challenge being the avoidance of ghosting artifacts in dynamic scenes. Recent methods have proposed using deep neural networks for deghosting. However, the methods typically rely on sufficient data with HDR ground-truths, which are difficult and costly to collect. In this work, to eliminate the need for labeled data, we propose SelfHDR, a self-supervised HDR reconstruction method that only requires dynamic multi-exposure images during training. Specifically, SelfHDR learns a reconstruction network under the supervision of two complementary components, which can be constructed from multi-exposure images and focus on HDR color as well as structure, respectively. The color component is estimated from aligned multi-exposure images, while the structure one is generated through a structure-focused network that is supervised by the color component and an input reference (\eg, medium-exposure) image. During testing, the learned reconstruction network is directly deployed to predict an HDR image. Experiments on real-world images demonstrate our SelfHDR achieves superior results against the state-of-the-art self-supervised methods, and comparable performance to supervised ones. 


## 2. Framework

<p align="center"><img src="selfhdr.png" width="95%"></p>
<p align="center">Overview of our proposed SelfHDR framework.</p>

- During training, we first construct color and structure components (*i.e.*, $Y_{color}$ and $Y_{stru}$), then take $Y_{color}$ and $Y_{stru}$ for supervising the HDR reconstruction network. Dotted lines with different colors represent different loss terms.
- During testing, the HDR reconstruction network can be used to predict HDR images from unseen multi-exposure images. 


    

## 3. Results

<p align="center"><img src="results.png" width="95%"></p>
