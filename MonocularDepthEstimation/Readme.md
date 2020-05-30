----------------------
## Name : Abhijit Mali
----------------------
## Notes 
---------------------------------------------------------------------------------------------------------------------------
1. Created 2 models one for Mask and other for Depth Map.
Mask model is modified architecture of Assignment 11 with Depthwise seperable convolutions
Depth Map model is inspired from UNET architecture along with that I have used Dice loss.
Model Params Count for Depth Mask model :- 492,947
Model Params Count for Mask model :- 197,558

For Depth Map used the following python notebook:
https://github.com/csharpshooter/EVA/blob/master/A15-PartB/A15_B_DM.ipynb

For Mask:
https://github.com/csharpshooter/EVA/blob/master/A15-PartB/A15_B_Mask.ipynb

Strategy of training:
First train both models for 64x64 image size, then 128x128 and then 224x224
I was able to train for all three size for depth map but was able to train only for i.e. 64 and 128 for masks model

I trained Depth Map model for for 12 epochs 4*64, 6*128, 224*2 and Mask model for 7 epochs 4*64, 128*3

I tried different losses for Mask prediction such as BCEWithLogits, MSELoss and SmoothL1Loss I found SmoothL1Loss gave me good output so used that.

For Depth Map prediction I tried DiceLoss, SSIM (Kornia) and SmoothL1 loss. Here Dice Loss gave me good output so used that

For training I created 2 models and did not merge them into one as it gave me flexibilty to train them on 2 different colab accounts
simultaneously. This helped me tweak them independently and tweak them as per their output.

I achieved IOU of 0.7 for both the DepthMap and Mask prediciton. I used IOU to monitor as accuracy as didnot get how to calculate accuracy in such cases

My dataset had all images with size 224x224. I did not use the 15-A assignment dataset. Created my own dataset as specified by you for the previous assignments. I used images of birds as foreground and landscapes as background. All these images were photgraphed by me :-)

Dataset link: contains images as well as zip files:
https://drive.google.com/drive/folders/1NXUBngY_z3M6d9DYs8KQZTXo9b7zIF8u?usp=sharing

I created zip file of images and then unzipped them on colab for training my models


    
---------------------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------------------------------
## Sample Outputs
--------------------
Order
1. FG+BG
2. Mask - Groundtruth
3. Mask predicted
4. Depth mask ground truth
5. Depth mask predicted

![O1](https://github.com/csharpshooter/EVA/blob/master/A15-PartB/images/SampleOutput.JPG)
![O2](https://github.com/csharpshooter/EVA/blob/master/A15-PartB/images/SampleOutput1.JPG)
![O3](https://github.com/csharpshooter/EVA/blob/master/A15-PartB/images/SampleOutput2.JPG)

---------------------------------------------------------------------------------------------------------------------------
## Test and Train, Loss and IOU Graphs

Mask
![MaskGraphs](https://github.com/csharpshooter/EVA/blob/master/A15-PartB/images/traintestgraphs_mask.png)

Depth Mask
![DepthMaskGraphs](https://github.com/csharpshooter/EVA/blob/master/A15-PartB/images/traintestgraphs_depthmasks.png)

---------------------------------------------------------------------------------------------------------------------------
## Model Summary

Depth Map Model:
----------------

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 64, 64]             880
       BatchNorm2d-2           [-1, 16, 64, 64]              32
              ReLU-3           [-1, 16, 64, 64]               0
            Conv2d-4           [-1, 16, 64, 64]           2,320
       BatchNorm2d-5           [-1, 16, 64, 64]              32
              ReLU-6           [-1, 16, 64, 64]               0
        DoubleConv-7           [-1, 16, 64, 64]               0
            Conv2d-8           [-1, 16, 64, 64]           2,320
              ReLU-9           [-1, 16, 64, 64]               0
           Conv2d-10           [-1, 16, 64, 64]           2,320
             ReLU-11           [-1, 16, 64, 64]               0
           Conv2d-12           [-1, 32, 64, 64]           4,640
             ReLU-13           [-1, 32, 64, 64]               0
           Conv2d-14           [-1, 32, 64, 64]           9,248
             ReLU-15           [-1, 32, 64, 64]               0
           Conv2d-16           [-1, 64, 64, 64]          18,496
             ReLU-17           [-1, 64, 64, 64]               0
           Conv2d-18           [-1, 64, 64, 64]          36,928
             ReLU-19           [-1, 64, 64, 64]               0
           Conv2d-20          [-1, 128, 64, 64]          73,856
             ReLU-21          [-1, 128, 64, 64]               0
           Conv2d-22          [-1, 128, 64, 64]         147,584
             ReLU-23          [-1, 128, 64, 64]               0
  ConvTranspose2d-24           [-1, 64, 64, 64]          73,792
             ReLU-25           [-1, 64, 64, 64]               0
  ConvTranspose2d-26           [-1, 64, 64, 64]          73,792
             ReLU-27           [-1, 64, 64, 64]               0
  ConvTranspose2d-28           [-1, 32, 64, 64]          18,464
             ReLU-29           [-1, 32, 64, 64]               0
  ConvTranspose2d-30           [-1, 32, 64, 64]          18,464
             ReLU-31           [-1, 32, 64, 64]               0
  ConvTranspose2d-32           [-1, 16, 64, 64]           4,624
             ReLU-33           [-1, 16, 64, 64]               0
  ConvTranspose2d-34           [-1, 16, 64, 64]           4,624
             ReLU-35           [-1, 16, 64, 64]               0
           Conv2d-36            [-1, 3, 64, 64]             435
      BatchNorm2d-37            [-1, 3, 64, 64]               6
             ReLU-38            [-1, 3, 64, 64]               0
           Conv2d-39            [-1, 3, 64, 64]              84
      BatchNorm2d-40            [-1, 3, 64, 64]               6
             ReLU-41            [-1, 3, 64, 64]               0
       DoubleConv-42            [-1, 3, 64, 64]               0
================================================================
Total params: 492,947
Trainable params: 492,947
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.19
Forward/backward pass size (MB): 48.16
Params size (MB): 1.88
Estimated Total Size (MB): 50.22
----------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------
Mask Model
----------
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 3, 64, 64]              27
            Conv2d-2           [-1, 64, 64, 64]             256
DepthwiseSeparableConv2d-3           [-1, 64, 64, 64]               0
              ReLU-4           [-1, 64, 64, 64]               0
            Conv2d-5           [-1, 64, 64, 64]             576
            Conv2d-6          [-1, 128, 64, 64]           8,320
DepthwiseSeparableConv2d-7          [-1, 128, 64, 64]               0
              ReLU-8          [-1, 128, 64, 64]               0
            Conv2d-9          [-1, 128, 64, 64]           1,152
           Conv2d-10          [-1, 128, 64, 64]          16,512
DepthwiseSeparableConv2d-11          [-1, 128, 64, 64]               0
      BatchNorm2d-12          [-1, 128, 64, 64]             256
           Conv2d-13          [-1, 128, 64, 64]           1,152
           Conv2d-14          [-1, 128, 64, 64]          16,512
DepthwiseSeparableConv2d-15          [-1, 128, 64, 64]               0
      BatchNorm2d-16          [-1, 128, 64, 64]             256
           Conv2d-17          [-1, 128, 64, 64]           1,152
           Conv2d-18          [-1, 128, 64, 64]          16,512
DepthwiseSeparableConv2d-19          [-1, 128, 64, 64]               0
           Conv2d-20          [-1, 128, 64, 64]           1,152
           Conv2d-21          [-1, 128, 64, 64]          16,512
DepthwiseSeparableConv2d-22          [-1, 128, 64, 64]               0
       BasicBlock-23          [-1, 128, 64, 64]               0
           Conv2d-24          [-1, 128, 64, 64]           1,152
           Conv2d-25           [-1, 64, 64, 64]           8,256
DepthwiseSeparableConv2d-26           [-1, 64, 64, 64]               0
             ReLU-27           [-1, 64, 64, 64]               0
           Conv2d-28           [-1, 64, 64, 64]             576
           Conv2d-29           [-1, 32, 64, 64]           2,080
DepthwiseSeparableConv2d-30           [-1, 32, 64, 64]               0
             ReLU-31           [-1, 32, 64, 64]               0
           Conv2d-32           [-1, 32, 64, 64]             288
           Conv2d-33           [-1, 32, 64, 64]           1,056
DepthwiseSeparableConv2d-34           [-1, 32, 64, 64]               0
      BatchNorm2d-35           [-1, 32, 64, 64]              64
           Conv2d-36           [-1, 32, 64, 64]             288
           Conv2d-37           [-1, 32, 64, 64]           1,056
DepthwiseSeparableConv2d-38           [-1, 32, 64, 64]               0
      BatchNorm2d-39           [-1, 32, 64, 64]              64
           Conv2d-40           [-1, 32, 64, 64]             288
           Conv2d-41           [-1, 32, 64, 64]           1,056
DepthwiseSeparableConv2d-42           [-1, 32, 64, 64]               0
           Conv2d-43           [-1, 32, 64, 64]             288
           Conv2d-44           [-1, 32, 64, 64]           1,056
DepthwiseSeparableConv2d-45           [-1, 32, 64, 64]               0
       BasicBlock-46           [-1, 32, 64, 64]               0
           Conv2d-47            [-1, 3, 64, 64]              27
           Conv2d-48           [-1, 64, 64, 64]             256
DepthwiseSeparableConv2d-49           [-1, 64, 64, 64]               0
             ReLU-50           [-1, 64, 64, 64]               0
           Conv2d-51           [-1, 64, 64, 64]             576
           Conv2d-52          [-1, 128, 64, 64]           8,320
DepthwiseSeparableConv2d-53          [-1, 128, 64, 64]               0
             ReLU-54          [-1, 128, 64, 64]               0
           Conv2d-55          [-1, 128, 64, 64]           1,152
           Conv2d-56          [-1, 128, 64, 64]          16,512
DepthwiseSeparableConv2d-57          [-1, 128, 64, 64]               0
      BatchNorm2d-58          [-1, 128, 64, 64]             256
           Conv2d-59          [-1, 128, 64, 64]           1,152
           Conv2d-60          [-1, 128, 64, 64]          16,512
DepthwiseSeparableConv2d-61          [-1, 128, 64, 64]               0
      BatchNorm2d-62          [-1, 128, 64, 64]             256
           Conv2d-63          [-1, 128, 64, 64]           1,152
           Conv2d-64          [-1, 128, 64, 64]          16,512
DepthwiseSeparableConv2d-65          [-1, 128, 64, 64]               0
           Conv2d-66          [-1, 128, 64, 64]           1,152
           Conv2d-67          [-1, 128, 64, 64]          16,512
DepthwiseSeparableConv2d-68          [-1, 128, 64, 64]               0
       BasicBlock-69          [-1, 128, 64, 64]               0
           Conv2d-70          [-1, 128, 64, 64]           1,152
           Conv2d-71           [-1, 64, 64, 64]           8,256
DepthwiseSeparableConv2d-72           [-1, 64, 64, 64]               0
             ReLU-73           [-1, 64, 64, 64]               0
           Conv2d-74           [-1, 64, 64, 64]             576
           Conv2d-75           [-1, 32, 64, 64]           2,080
DepthwiseSeparableConv2d-76           [-1, 32, 64, 64]               0
             ReLU-77           [-1, 32, 64, 64]               0
           Conv2d-78           [-1, 32, 64, 64]             288
           Conv2d-79           [-1, 32, 64, 64]           1,056
DepthwiseSeparableConv2d-80           [-1, 32, 64, 64]               0
      BatchNorm2d-81           [-1, 32, 64, 64]              64
           Conv2d-82           [-1, 32, 64, 64]             288
           Conv2d-83           [-1, 32, 64, 64]           1,056
DepthwiseSeparableConv2d-84           [-1, 32, 64, 64]               0
      BatchNorm2d-85           [-1, 32, 64, 64]              64
           Conv2d-86           [-1, 32, 64, 64]             288
           Conv2d-87           [-1, 32, 64, 64]           1,056
DepthwiseSeparableConv2d-88           [-1, 32, 64, 64]               0
           Conv2d-89           [-1, 32, 64, 64]             288
           Conv2d-90           [-1, 32, 64, 64]           1,056
DepthwiseSeparableConv2d-91           [-1, 32, 64, 64]               0
       BasicBlock-92           [-1, 32, 64, 64]               0
           Conv2d-93            [-1, 3, 64, 64]           1,728
================================================================
Total params: 197,558
Trainable params: 197,558
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.19
Forward/backward pass size (MB): 220.28
Params size (MB): 0.75
Estimated Total Size (MB): 221.22
----------------------------------------------------------------

