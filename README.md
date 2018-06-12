# AMNet: Memorability Estimation with Attention
A PyTorch implementation of our paper [AMNet: Memorability Estimation with Attention](https://arxiv.org/abs/1804.03115)
by Jiri Fajtl, Vasileios Argyriou, Dorothy Monekosso and Paolo Remagnino. This paper will be presented 
at [CVPR 2018](http://cvpr2018.thecvf.com/).
 

## Installation
The development and evaluation was done on the following configuration:
### System configuration
- Platform :  Linux-4.13.0-38-generic-x86_64-with-Ubuntu-16.04-xenial
- Display driver : NVRM version: NVIDIA UNIX x86_64 Kernel Module  384.111  Tue Dec 19 23:51:45 PST 2017
				GCC version:  gcc version 5.4.0 (Ubuntu 5.4.0-6ubuntu1~16.04.9)
- GPU: NVIDIA Titan Xp
- CUDA:  8.0.61
- CUDNN: 6.0.21

### Python packages
- Python: 3.5.2
- PyTorch:  0.2.0_2
- Torchvision:  0.1.9
- NumPy:  1.14.2
- OpenCV:  3.2.0
- PIL:  1.1.7

There are no explicit version requirements for any of the components and the AMNet is expected to run with other configurations too,
however, it is important to note that results obtained with different setup may slightly differ from our publication. We found that 
with PyTorch version 0.3.0.post4 the average Spearman's rank correlation over the five LaMem splits was RC=0.67641 while with the 
version 0.2.0_2  RC=0.67666 (published).


## Datasets
The AMNet was evaluated on two datasests, [LaMem](http://memorability.csail.mit.edu/download.html) and 
[SUN memorability](http://web.mit.edu/phillipi/Public/WhatMakesAnImageMemorable/). The original SUN memorability dataset
was processed to the format identical with LaMem. Both datasets can be downloaded bu running the following commands. You will need
~3GB space on your driver (2.7GB for LaMem and 280MB for SUN).
```
cd datasets
../download.sh lamem_url.txt
../download.sh sun_memorability_url.txt
```
You can also use wild card '*.txt' to download them.

## Trained Models
In order to quikly verify the published results or use the AMNet for your own application you can download  
models fully trained on the LaMem and SUN datatests. You can download all by running the following command. 
You will need ~11GB space on your drive. 
```
cd data
../download.sh *.txt
```

Alternatively you can download weights for each test case individually.
```
cd data
../download.sh lamem_weights_urls.txt
../download.sh sun_weights_urls.txt
```
The models will be stored in the 'data' directory, one for each split.

| Model | size |
| ---- | :----:|
| lamem_ResNet50FC_lstm3_train_* | 822MB |
| lamem_ResNet50FC_lstm3_noatt_train_* | 822MB |
| lamem_ResNet101FC_lstm3_train_* | 1.2GB |
| sun_ResNet50FC_lstm3_train_* | 4GB |
| sun_ResNet50FC_lstm3_noatt_train_* | 4GB |

Where `ResNet*` signifies name of CNN model used for features extraction, `noatt` stands for 'no visual attention' and `lstm3` a LSTM sequence with three steps. 

## Evaluation
Evaluation on the LaMem and SUN datasets was done according to protocols suggested by authors of the datasets.
The LaMem was evaluated on 5 and the SUN on 25 train/test splits. Each evaluation was done twice, with the attention
enabled and disabled. To run the LaMem evaluation please fisrt download the LaMem dataset [Datasets](#Datasets) and 
the trained models [Trained Models](#Trained Models) and then run
```
python3 main.py --test --dataset lamem --cnn ResNet50FC --test-split 'test_*'
```
In order to run the evaluation without the attention specifiy `--att-off` argument
```
python3 main.py --test --dataset lamem --cnn ResNet50FC --test-split 'test_*' --att-off'
```

## Predicting memorability of images
If you wish to estimate memorability for your images you have two options, process all 
images in a given directory or create a csv file with a list of images to process. In both cases
you need to specify file with the model weights. To predict memorability of all images in directory 
run this command
```
python3 main.py --cnn ResNet50FC --model-weights data/lamem_ResNet50FC_lstm3_train_5/weights_35.pkl --eval-images images/high
```
Memorability of each image will be printed on the stdout. If you want to save the memorabilities to a csv file specify
argument `--csv-out <filename.txt>`
```
python3 main.py --cnn ResNet50FC --model-weights data/lamem_ResNet50FC_lstm3_train_5/weights_35.pkl --eval-images images/high --csv-out memorabilities.txt
```

Attention maps for each LSTM step can be stored as a jpg image for each input image by specifying output path `--att-maps-out <out_dir>`
```
python3 main.py --cnn ResNet50FC --model-weights data/lamem_ResNet50FC_lstm3_train_5/weights_35.pkl --eval-images images/high --att-maps-out att_maps
```

Here is an example of the attention map image. It includes the original image and one image for each LSTM step with the attention map
shown as a heatmap overlay.
![img1](att_maps/021_att.jpg)

## Training
To train the AMNet you need to follow these steps

- select CNN front end for image features extraction. Available models are ResNet18FC, ResNet50FC, ResNet101FC and VGG16FC. 
- select lamem or sun dataset.
- specify training and validation splits. Note that the SUN memorability dataset doesn't come with validation split, thus the test split need to be used.
- optionally you can set the batch sizes, gpu id, disable the visual attention and other. Please run ```main.py --help``` to see other options.

```
python3 main.py --train-batch-size 222 --test-batch-size 222 --cnn ResNet50FC --dataset lamem --train-split train_1 --val-split val_1
```

To see other command line arguments please run
```
python3 main.py --help
```
or see [main.py](main.py). If you want to experiment with other parameters the best place to go is [config.py](config.py).


## Cite
If you use this code or reference our paper in your work please cite this publication.
```
@inproceedings{fajtl2018amnet,
  title={AMNet: Memorability Estimation with Attention},
  author={Fajtl, Jiri and Argyriou, Vasileios and Monekosso, Dorothy and Remagnino, Paolo},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={6363--6372},
  year={2018}
}
```

