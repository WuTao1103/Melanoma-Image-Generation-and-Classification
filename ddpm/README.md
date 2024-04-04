 DDPM: Denoising Diffusion Probabilistic Models in Pytorch
---DDPM: Denoising Diffusion Probabilistic Models in pytorch

### Contents
1. [Required Environment Environment](# Required Environment)
3. [File Download Download](#File Download)
4. [Prediction step How2predict](#Prediction step)
5. [Training step How2train](#Training step)
6. [Reference Reference](#Reference)

## Required environment
pytorch==1.7.0 
pytorch==1.2.0 can't load weights over 2G under windows, it can't work properly, not recommended.

## Prediction steps
### a. Use pre-trained weights
1. Unzip the library after downloading, run predict.py directly, click enter in the terminal, you can generate images, the generated images are located in results/predict_out/predict_1x1_results.png, results/predict_out/predict_5x5_ results.png.    
### b. Use your own trained weights 
1. Follow the training steps.    
2. Inside the dcgan.py file, modify model_path to correspond to the trained file in the following section; **model_path corresponds to the weights file under the logs folder**.   
 ```python
_defaults = {
    #-----------------------------------------------#
    # model_path points to the weights file in the logs folder
    #-----------------------------------------------#
    "model_path" : 'model_data/Diffusion_Flower.pth',
    #-----------------------------------------------#
    # Settings for the number of convolution channels
    #-----------------------------------------------#
    "channel" : 128, ###
    #-----------------------------------------------#
    # Settings for input image size
    #-----------------------------------------------#
    "input_shape" : (64, 64), ###
    #---------------------------------------------------------------------#
    # betas related parameters
    #---------------------------------------------------------------------#
    "schedule" : "linear",
    "num_timesteps" : 1000,
    "schedule_low" : 1e-4,
    "schedule_high" : 0.02,
    #-------------------------------#
    # Whether to use Cuda
    # No GPU can be set to False
    #-------------------------------#
    "cuda" : True, ##
}
```
3. Run predict.py and click enter in the terminal to generate the images, the generated images are located in results/predict_out/predict_1x1_results.png, results/predict_out/predict_5x5_results.png.    

## Training steps
1. Before training, put the expected image files in the datasets folder.
2. Run txt_annotation.py under the root directory to generate train_lines.txt, make sure that there is a file path content inside train_lines.txt.  
3. Run the train.py file for training, the images generated during the training process can be viewed in the results/train_out folder.
