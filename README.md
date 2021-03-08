# Generating Object Stamps

This is the official code base for the paper:  [Generating object Stamps](https://arxiv.org/pdf/2001.02595.pdf). 

Presented in the AI for Content Creation CVPR workshop 2020 [(AI4CCW)](http://visual.cs.brown.edu/workshops/aicc2020/).

This code has been trained on python 3.7 on a ubuntu system with 2xGPU (GTX 1080ti).

![Alt text](./Images/teaser.png?raw=true "Teaser")

### Install requirements

Create a conda environment and install requirements: 

 ```bash requirements.sh```

Activate environment:

 ``` conda activate stamps ```

### Create data and train
To run the code, Follow the steps bellow:  

+ Download the vgg_16.ckpt model from: https://drive.google.com/open?id=1jHkrz1Usp9JHylS6gtCcHuENcOcKbVAm

+ Create the training and validation data using the command below and replacing with the correct paths (Install pycocotools first ```conda install -c conda-forge pycocotools```).: 

```python create_data.py Yo```

and 

```python create_data.py --path /home/yam28/Documents/phdYoop/datasets/COCO --dataset giraffe --mode val```


You are now ready to start training:

+ Training the mask generation part can be done via: 

```python mask_gen.py --dataset giraffe --data ./dataset --path /home/yam28/Documents/phdYoop/datasets/COCO```

The argument 'data' is where the generated data above is stored. 

The argument 'path' being the path for the coco dataset.

The argument 'dataset' specifies the name of the dataset. 

The training logs and checkpoints will be stored under the folder 'train_log'.

+ Training the texture generation part can be done via: 

```python rgb_gen.py --dataset giraffe --data ./dataset --path /home/yam28/Documents/phdYoop/datasets/COCO --load ./vgg_16.ckpt```

The training logs and checkpoints will be stored under the folder 'train_log'.

+ You can chek intermediate results while training using tensorboard. First cd to the checkpoint directory and use:

``` tensorboard --logdir=.```

### Inference
The process for inference is visualized below: 

![Alt text](./Images/inference.png?raw=true "Teaser")

When the training is finished, the graphs for both mask and texture generation are frozen and automatically saved under 'frozen_model.pb' in the corresponding folders in 'train_log'.
To visualize results on new images using bounding boxes from the validation set, run the command below with the appropriate arguments, in my case this would be:

``` python Inference_Instance_generation_grid.py --mask_model ./train_log/giraffe/mask_gen/20200113-113913/frozen_model.pb --rgb_model ./train_log/giraffe/rgb_gen/20200113-114121/frozen_model.pb --mask_folder ./dataset/val/giraffe --bg_folder ./bg_data/savannah```

where the arguments are described below: 

'mask_model': the path for 'frozen_model.pb' corresponding to the mask generation.

'rgb_model': the path for 'frozen_model.pb' corresponding to the texture generation.

'mask_folder': is the validation folder created by ```create_data.py```.

'bg_folder': The path to a folder containing the background images on top which you would like to generate stamps. 
