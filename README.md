# StyleGAN2-using-pytorch-without-CUDA-error
<h2>Implementation of StyleGAN2-ada using Pytorch without any version related errors and cuda errors.</h2>

##Implementing StyleGAN2 with PyTorch 1.7.1 in Google Colab
This guide provides instructions for setting up and running StyleGAN2 using PyTorch 1.7.1 in Google Colab. Follow these steps to ensure compatibility and successful execution of your StyleGAN2 project.

##Navigate to Google Colab and start a new notebook.
###Install Miniconda:
To manage the specific version dependencies, it's recommended to install Miniconda in your Colab environment. Use the following commands to install Miniconda:

!wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh 

!bash Miniconda3-latest-Linux-x86_64.sh -bfp /usr/local

import os

os.environ['PATH'] = "/usr/local/bin:" + os.environ['PATH']

!conda update -n base -c defaults conda -y

##Create a Virtual Environment with Python 3.7:
With Miniconda installed, create a new virtual environment using Python 3.7:

!conda create -n myenv python=3.7 -y

Activate the newly created environment:
!conda run -n myenv

##Install Dependencies:
Install PyTorch 1.7.1, torchvision 0.8.2, torchaudio 0.7.2, and cudatoolkit 11.0 in the virtual environment:

!conda install -n myenv pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch -y

##Clone NVIDIA's StyleGAN2-ADA-PyTorch Repository:
!git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git 

Navigate to the cloned repository directory:
%cd stylegan2-ada-pytorch

##Install all the required Libraries
click,tqdm,requests,psutil,scipy

##Prepare the Dataset:
Use the dataset_tool.py script to prepare your dataset for training:

!conda run -n myenv python dataset_tool.py --source /path/to/your/dataset --dest /content/drive/MyDrive/custom_dataset/input_set --width=256 --height=256 --resize-filter=box

##Train the Model:
Initiate the training process with the train.py script:

!conda run -n myenv python train.py --data /content/drive/MyDrive/custom_dataset/input_set --outdir /content/drive/MyDrive/custom_dataset/results

Use a small kimg number for fast results at low quality. This is useful for quick iterations when fine-tuning your model's parameters.
For better quality results, retain the default kimg number. Note that while this will yield higher quality, the training will take longer.
In the above code provided, the kimg value is set to its default setting.
If you want to customize the kimg value, update your training command accordingly. For example, to set kimg to 1500, use the following command:
!conda run -n myenv python train.py --data /content/drive/MyDrive/custom_dataset/input_set --outdir /content/drive/MyDrive/custom_dataset/results --kimg=1500

##Generate Images:
After training, use the generated pickle files with the generator.py script to create images:

!conda run -n myenv python generate.py --outdir=/content/drive/MyDrive/custom_dataset/output --trunc=1 --seeds=85,265,297,849 --network=/content/drive/MyDrive/custom_dataset/result/network-snapshot-000000.pkl
