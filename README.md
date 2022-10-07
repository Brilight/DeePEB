# DeePEB

### Intro

PEB simulation acts as the bridge between the aerial image and the final resist profile in lithography simulation. To accelerate the PEB simulation without sacrificing accuracy, we propose DeePEB, a neural PDE solver. 

We construct DeePEB based on the observation of the physical essence of PEB: most of the dynamic information of the PEB process is contained in low-frequency modes of related reactants, and the high-frequency information affects the local features. So we combine both neural operator and customized convolution operations for learning the solution operator of PEB. Our algorithm is validated with an industry-strength software S-Litho under real manufacturing conditions, exhibiting high efficiency and accuracy.

This repo provides the source code of DeePEB, created by Qipan on 2022.1.18, project set up on 2021.9.

For some privacy and copyright concerns, the dataset and trained models are partially provided; more training and test data are accessible upon proper requirements.

Detailed algorithms and details can be found in our [paper](https://github.com/Brilight/DeePEB/blob/main/ICCAD2022-468-DeePEB.pdf), which can be cited as (going to update):



### Requirements:

0. Python >3.5

1. Pytorch>=1.9.0 [CUDA > 11.0]

2. Numpy

3. tqdm

4. Other common packages in python, including csv, os, math, matplotlib, etc. 


### Usage

1. 
Just run the jupyter notebook with proper data form: 3d array in the zyx form

The test data can be found in Datas.Testdata 

2.
The .py file can be exported from the jupyter notebook file. 

We provide the running code for DeePEB_v1 only, while the source code of the other learning-based methods can be generated in the same manner.
