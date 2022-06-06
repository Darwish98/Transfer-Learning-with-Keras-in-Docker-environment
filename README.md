# Transfer-Learning-with-Keras-in-Docker-environment

A step-by-step tutorial on applying fixed feature extraction transfer learning, with the ResNet50 on the Cifar 10, Cifar 100, and Fashion MNIST datasets. Training with different datasets will make it possible to compare transfer learning results on datasets of different sizes and complexity. This tutorial also shows how to do fine tuning which is an optional step to increase accuracy.

## Table of contents

- [Motivation](#motivation)
- [A summary of result](#asummaryofresult)
- [Getting Started](#setup)

## Motivation

This tutorial shows how to apply transfer learning with Keras on TensorFlow by implementing ResNet50 model and comparing it to training a CNN from scratch. Different dataset as Cifar and FashionMnist will be used with the implemented models to evaluate Transfer Learning Capabilities with ResNet50.

## A summary of result

### Cifar 10

| Model                                | Accuracy before FT | Loss before FT | Accuracy after FT | Loss after FT |
| ------------------------------------ | ------------------ | -------------- | ----------------- | ------------- |
| A CNN from Scratch                   | 0.665              | 1.138          | ~                 | ~             |
| ResNet50 Transfer Learning with SGD  | 0.838              | 0.524          | 0.857             | 0.458         |
| ResNet50 Transfer Learning with Adam | 0.831              | 0.744          | 0.922             | 0.428         |
| Vision Transformer (ViT-B32)         | ~                  | ~              | 0.978             | 0.565         |

### Cifar 100

| Model                                | Accuracy before FT | Loss before FT | Accuracy after FT | Loss after FT |
| ------------------------------------ | ------------------ | -------------- | ----------------- | ------------- |
| A CNN from Scratch                   | 0.337              | 3.027          | ~                 | ~             |
| ResNet50 Transfer Learning with SGD  | 0.614              | 1.422          | 0.650             | 1.345         |
| ResNet50 Transfer Learning with Adam | 0.578              | 1.95           | 0.644             | 2.207         |
| Vision Transformer (ViT-B32)         | ~                  | ~              | 0.890             | 1.453         |

### Fashion Mnist

| Model                                                     | Accuracy before FT | Loss before FT | Accuracy after FT | Loss after FT |
| --------------------------------------------------------- | ------------------ | -------------- | ----------------- | ------------- |
| A CNN from Scratch                                        | 0.898              | 0.311          | ~                 | ~             |
| ResNet50 Transfer Learning with SGD and modified ResNet   | 0.685              | 0.827          | 0.761             | 0.626         |
| ResNet50 Transfer Learning with SGD and modified dataset  | 0.914              | 0.243          | 0.919             | 0.234         |
| ResNet50 Transfer Learning with Adam and modified ResNet  | 0.811              | 0.525          | 0.897             | 0.316         |
| ResNet50 Transfer Learning with Adam and modified dataset | 0.901              | 0.295          | 0.928             | 0.274         |

The accuracy was improved allot by transfer learning on all the datasets. The best achived accuracy on the cifar 10 and cifar 100 is from the fine tuned ViT. Also the results from applying transfer learning on ResNet50 is much better compared to trained from scratch CNN.

All the ResNet50 are trained in a docker environment and later on a more detalied tutorial will explain how to implement it.

The ViT was run with the original implementation from the ViT reaserch paper. You can run on the browser by using google colab from this link proposed on the reserach paper. Checkout these links for more: <br />
https://colab.research.google.com/github/google-research/vision_transformer/blob/main/vit_jax.ipynb <br />
https://github.com/google-research/vision_transformer

## Getting Started

## Docker setup

- [Install docker](#dockerinstall)
- [Build Tensorflow Image](#dockerbuild)
- [Build and run container](#dockerrun)

### Install docker

The first step is to install docker on your machine by following the tutorial from the offical docker website.
https://docs.docker.com/engine/install/ubuntu/

### Build Tensorflow Image

     1. Clone this repository by opening terminal and typing the following command:
     $git clone https://github.com/Darwish98/Transfer-Learning-with-Keras-in-Docker-environment.git

> **_NOTE:_** In case you want add more packges to the dockerfile follow step I to IV.

The dockerfiles in this repository are from the offical tensorflow repository.
Modification has been done on the docker files and they have all the important packages to run tesnorflow keras. They also have packages for visualization as matplotlib.
Step I - III is only if you want to add your own packages, if not go to step 2.

      I. Go to /docker/dockerfiles

      II. Open gpu-jupyter.Dockerfile if you want to run jupyter notebook.
          Open gpu.Dockerfile if you want to run tensorflow in terminal.

      III. Once your have chosen a dockerfile from step two open it.

      IV.  Add your packages with "pip" keyword before it. When done save and close the file.

>

      2. Open a new terminal and cd to the docker folder from this repository.

      3. Run the following command to build the image from the dockerfile:

         A) In case you want to run tensorflow in terminal run the following command
         $ docker build -f ./dockerfiles/gpu.Dockerfile -t tf .
         B) In case you want to run Jupyter Notebook run the following command
         $ docker build -f ./dockerfiles/gpu-jupyter.Dockerfile -t tf .

      4. Run the following command to build the container:

         # In case you built a tensorflow in terminal image (option A) run the following command:
         $ docker run --runtime=nvidia -u $(id -u):$(id -g) -v $(pwd):/my-devel -it tf

         # In case you built a Jupyter Notebook image (option B) run the following command:
         $docker run --user $(id -u):$(id -g) -p 8888:8888 -v $(PWD):/tf/notebooks -it tf

#### Step 2-5 (in case you chose tensorflow in terminal) are shown in the following Gif:

![Alt text](https://media0.giphy.com/media/nIB2oaRbyWXiQNCTPD/giphy.gif?cid=790b7611b7ee36600e5c419ba0710d4183f156a6f4fcdd05&rid=giphy.gif "Optional title")

#### Step 2-5 (in case you chose Jupyter Notebook) are shown in the following Gif:

![Alt text](https://media0.giphy.com/media/9hFOLPcaUdOI9ZQYcX/giphy.gif?cid=790b761128a7745edcba93d34512a7ef924d72e1b3544a16&rid=giphy.gif "Optional title")
