----------------------
## Name : Abhijit Mali
----------------------
## Notes 
---------------------------------------------------------------------------------------------------------------------------
1. Chose 'drone' as an object for detection as it is not present in the COCO dataset. This model will work for most of the drones in the market as I have collected image dataset for all kinds of drones.
#### Images : https://github.com/csharpshooter/DeepLearning/tree/master/CustomObjectDetectionUsingYoloV3/data/customdata/images

2. Trained for 300 epochs as well as 1000 epochs. After training for 1000 epochs found that the small discrepancies were gone and objects were being detetcted correctly. Did not know that if we were supposed to train for 1000 epochs so have added outputs for 300 epochs training. Following are the graphs for both.

## 300 epochs Graphs
![300epochs](https://github.com/csharpshooter/DeepLearning/blob/master/CustomObjectDetectionUsingYoloV3/images/300%20epochs%20tgraphs.png)

## 1000 epochs Graphs
![1000epochs](https://github.com/csharpshooter/DeepLearning/blob/master/CustomObjectDetectionUsingYoloV3/images/1000%20epochs%20tboard%20graphs.png)

3. Collected 563 files for dataset from different websites like unsplash, shutterstock, depositphotos for all types of drones and trained on them.

## Train 
![TrainImgfromdataset](https://github.com/csharpshooter/DeepLearning/blob/master/CustomObjectDetectionUsingYoloV3/images/trainsample.png)

## Test 
![TestImgfromdataset](https://github.com/csharpshooter/DeepLearning/blob/master/CustomObjectDetectionUsingYoloV3/images/test%20sample.png)

4. For inference I have targeted three DJI Inc. drones 'DJI Spark', 'Phantom' and 'Mavic'  videos and combined their clips and uploaded to youtube. I did not create 3 different classes for them as I thought they come under the same drone generic class. 
#### Youtube Video Link: https://www.youtube.com/watch?v=E4iT1pVvpJQ

5. Link to 1000 epochs trained notebok file: https://github.com/csharpshooter/DeepLearning/blob/master/CustomObjectDetectionUsingYoloV3/A13-B-1000-Epochs.ipynb
---------------------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------------------------------
## Model Summary

Model Summary: 225 layers, 6.25733e+07 parameters, 6.25733e+07 gradients

---------------------------------------------------------------------------------------------------------------------------
## Logs

Namespace(accumulate=4, adam=False, batch_size=10, bucket='', cache_images=True, cfg='cfg/yolov3-custom.cfg', data='data/customdata/custom.data', device='', epochs=300, evolve=False, img_size=[512], multi_scale=False, name='', nosave=True, notest=False, rect=False, resume=False, single_cls=True, weights='weights/yolov3-spp-ultralytics.pt')
Using CUDA device0 _CudaDeviceProperties(name='Tesla P100-PCIE-16GB', total_memory=16280MB)

2020-04-25 11:09:36.206063: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
Run 'tensorboard --logdir=runs' to view tensorboard at http://localhost:6006/
Model Summary: 225 layers, 6.25733e+07 parameters, 6.25733e+07 gradients
Caching labels (563 found, 0 missing, 0 empty, 0 duplicate, for 563 images): 100% 563/563 [00:00<00:00, 10660.15it/s]
Caching images (0.3GB): 100% 563/563 [00:02<00:00, 262.14it/s]
Caching labels (563 found, 0 missing, 0 empty, 0 duplicate, for 563 images): 100% 563/563 [00:00<00:00, 9838.81it/s]
Caching images (0.2GB): 100% 563/563 [00:02<00:00, 230.54it/s]
Image sizes 512 - 512 train, 512 test
Using 2 dataloader workers
Starting training for 300 epochs...

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     0/299     7.38G      6.03      3.37         0       9.4         7       512: 100% 57/57 [00:39<00:00,  1.45it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:17<00:00,  3.28it/s]
                 all       563       597         1   0.00168     0.244   0.00334

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     1/299     12.1G      4.25      1.79         0      6.04         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:10<00:00,  5.29it/s]
                 all       563       597     0.184      0.76     0.469     0.296

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     2/299     12.1G      3.97      1.39         0      5.36         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:13<00:00,  4.25it/s]
                 all       563       597     0.326     0.732     0.467     0.451

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     3/299     12.1G      3.93      1.19         0      5.11         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:11<00:00,  4.84it/s]
                 all       563       597     0.634      0.75      0.69     0.688

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     4/299     12.1G      3.54      1.15         0      4.69         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:10<00:00,  5.25it/s]
                 all       563       597     0.655     0.849     0.837      0.74

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     5/299     12.1G      3.32      1.12         0      4.44         5       512: 100% 57/57 [00:37<00:00,  1.51it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:10<00:00,  5.24it/s]
                 all       563       597     0.711     0.807     0.804     0.756

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     6/299     12.1G      3.08      1.03         0      4.11         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:10<00:00,  5.55it/s]
                 all       563       597     0.855     0.863     0.884     0.859

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     7/299     12.1G      2.86         1         0      3.86         9       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:10<00:00,  5.43it/s]
                 all       563       597     0.893     0.856     0.896     0.874

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     8/299     12.1G      3.16     0.958         0      4.12        17       512:  77% 44/57 [00:29<00:08,  1.52it/s]
Model Bias Summary:    layer        regression        objectness    classification
                          89      -0.09+/-0.20      -5.35+/-0.56       4.11+/-0.01 
                         101       0.09+/-0.23      -6.00+/-0.33       4.13+/-0.00 
                         113       0.17+/-0.29      -6.11+/-0.27       4.09+/-0.01 
     8/299     12.1G       3.1     0.966         0      4.07         4       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:10<00:00,  5.46it/s]
                 all       563       597     0.904     0.799     0.828     0.848

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     9/299     12.1G      2.86     0.894         0      3.76         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:10<00:00,  5.61it/s]
                 all       563       597     0.873     0.896     0.906     0.884

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    10/299     12.1G      2.66     0.844         0      3.51         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:10<00:00,  5.38it/s]
                 all       563       597     0.742     0.807     0.821     0.773

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    11/299     12.1G      2.65     0.912         0      3.56         9       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:10<00:00,  5.49it/s]
                 all       563       597     0.807     0.869     0.897     0.837

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    12/299     12.1G      2.72     0.858         0      3.58         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:10<00:00,  5.55it/s]
                 all       563       597     0.925     0.847      0.91     0.885

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    13/299     12.1G      2.52     0.804         0      3.32         4       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:10<00:00,  5.65it/s]
                 all       563       597     0.933     0.892     0.928     0.912

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    14/299     12.1G       2.5      0.77         0      3.27         4       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:10<00:00,  5.55it/s]
                 all       563       597     0.928      0.91     0.931     0.919

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    15/299     12.1G       2.2     0.771         0      2.97         9       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.74it/s]
                 all       563       597     0.921     0.891     0.922     0.906

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    16/299     12.1G      2.58     0.732         0      3.31         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:10<00:00,  5.65it/s]
                 all       563       597     0.922      0.91     0.935     0.916

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    17/299     12.1G      2.34      0.76         0       3.1         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.74it/s]
                 all       563       597     0.959     0.891     0.936     0.924

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    18/299     12.1G      2.24      0.72         0      2.96         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:10<00:00,  5.52it/s]
                 all       563       597     0.894     0.925     0.938     0.909

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    19/299     12.1G      2.31      0.74         0      3.05         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:10<00:00,  5.61it/s]
                 all       563       597     0.914     0.925     0.937      0.92

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    20/299     12.1G      2.27     0.729         0         3         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.78it/s]
                 all       563       597     0.946     0.916      0.94     0.931

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    21/299     12.1G      2.12     0.676         0       2.8         9       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.78it/s]
                 all       563       597      0.91     0.915     0.935     0.912

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    22/299     12.1G      2.17     0.646         0      2.82         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.79it/s]
                 all       563       597     0.948     0.899     0.932     0.923

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    23/299     12.1G      2.05     0.663         0      2.71         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.77it/s]
                 all       563       597     0.937     0.913     0.937     0.925

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    24/299     12.1G      1.94     0.672         0      2.62         4       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.76it/s]
                 all       563       597     0.919     0.915     0.926     0.917

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    25/299     12.1G      2.18     0.668         0      2.85         8       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597     0.942     0.928     0.949     0.935

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    26/299     12.1G      2.11     0.669         0      2.78         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.81it/s]
                 all       563       597     0.947     0.925     0.941     0.936

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    27/299     12.1G      2.16     0.682         0      2.84         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.77it/s]
                 all       563       597     0.942     0.905     0.932     0.923

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    28/299     12.1G      1.95     0.687         0      2.64         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.75it/s]
                 all       563       597     0.953     0.938     0.959     0.945

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    29/299     12.1G      1.88     0.634         0      2.51         8       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.79it/s]
                 all       563       597      0.96     0.928     0.962     0.944

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    30/299     12.1G      1.96     0.633         0       2.6         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.74it/s]
                 all       563       597     0.949     0.951     0.966      0.95

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    31/299     12.1G      2.05     0.593         0      2.64         4       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.77it/s]
                 all       563       597     0.949     0.926     0.956     0.937

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    32/299     12.1G      1.89     0.615         0      2.51         3       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.77it/s]
                 all       563       597      0.97     0.925     0.961     0.947

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    33/299     12.1G      2.17     0.636         0      2.81         8       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.77it/s]
                 all       563       597     0.949      0.95     0.959      0.95

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    34/299     12.1G      1.85       0.6         0      2.45         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597     0.949      0.94     0.957     0.944

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    35/299     12.1G      1.88     0.614         0      2.49        11       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.81it/s]
                 all       563       597     0.961      0.93     0.956     0.945

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    36/299     12.1G      1.86     0.595         0      2.45         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.79it/s]
                 all       563       597     0.972     0.935     0.969     0.953

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    37/299     12.1G      1.84     0.608         0      2.44         9       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.79it/s]
                 all       563       597     0.962     0.931     0.963     0.946

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    38/299     12.1G      1.81     0.574         0      2.39         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.78it/s]
                 all       563       597     0.975     0.946     0.963      0.96

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    39/299     12.1G       1.8      0.59         0      2.39         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.79it/s]
                 all       563       597     0.966     0.942     0.965     0.953

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    40/299     12.1G      1.77     0.557         0      2.33         8       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.79it/s]
                 all       563       597     0.976     0.933     0.965     0.954

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    41/299     12.1G      1.65     0.572         0      2.22         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.84it/s]
                 all       563       597     0.967     0.933     0.955      0.95

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    42/299     12.1G      1.81     0.543         0      2.35         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.81it/s]
                 all       563       597      0.97     0.948     0.966     0.959

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    43/299     12.1G      1.69     0.527         0      2.21         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.81it/s]
                 all       563       597     0.955     0.941     0.962     0.948

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    44/299     12.1G      1.71     0.553         0      2.27         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.82it/s]
                 all       563       597     0.979     0.953      0.97     0.966

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    45/299     12.1G      1.78     0.557         0      2.34         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597     0.985     0.948     0.972     0.966

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    46/299     12.1G      1.81     0.542         0      2.35         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.81it/s]
                 all       563       597     0.975     0.941      0.97     0.958

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    47/299     12.1G      1.57     0.508         0      2.08         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.82it/s]
                 all       563       597     0.967     0.941     0.964     0.954

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    48/299     12.1G      1.66      0.54         0       2.2         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.82it/s]
                 all       563       597     0.973     0.917     0.948     0.945

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    49/299     12.1G       1.9     0.551         0      2.45         9       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.81it/s]
                 all       563       597     0.963     0.956     0.972      0.96

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    50/299     12.1G      1.74     0.513         0      2.25         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.83it/s]
                 all       563       597     0.976     0.946     0.968     0.961

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    51/299     12.1G      1.63     0.523         0      2.16         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.81it/s]
                 all       563       597     0.965     0.941     0.962     0.953

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    52/299     12.1G      1.69     0.546         0      2.24         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.84it/s]
                 all       563       597     0.974     0.946     0.967      0.96

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    53/299     12.1G      1.69     0.536         0      2.23         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.83it/s]
                 all       563       597     0.978     0.958     0.975     0.968

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    54/299     12.1G      1.55     0.536         0      2.08         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597     0.975     0.966     0.978     0.971

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    55/299     12.1G      1.63     0.512         0      2.14         4       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.82it/s]
                 all       563       597     0.978     0.949     0.973     0.963

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    56/299     12.1G      1.67     0.514         0      2.18         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597     0.984     0.931     0.962     0.957

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    57/299     12.1G       1.6     0.542         0      2.14        10       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.82it/s]
                 all       563       597     0.984     0.933      0.97     0.958

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    58/299     12.1G      1.59     0.486         0      2.08         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.81it/s]
                 all       563       597     0.982      0.95     0.977     0.966

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    59/299     12.1G      1.77     0.473         0      2.24         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.81it/s]
                 all       563       597     0.982      0.94     0.967     0.961

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    60/299     12.1G      1.53     0.514         0      2.04         3       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.82it/s]
                 all       563       597     0.984     0.948     0.973     0.966

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    61/299     12.1G      1.69      0.51         0       2.2         9       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.83it/s]
                 all       563       597     0.967     0.948     0.976     0.958

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    62/299     12.1G      1.64     0.499         0      2.14         4       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.82it/s]
                 all       563       597     0.975      0.96     0.977     0.967

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    63/299     12.1G      1.51     0.532         0      2.04         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.84it/s]
                 all       563       597     0.984     0.947     0.972     0.965

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    64/299     12.1G      1.38     0.475         0      1.85         3       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.83it/s]
                 all       563       597     0.978      0.95     0.967     0.964

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    65/299     12.1G      1.59     0.498         0      2.08         9       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.81it/s]
                 all       563       597     0.979     0.949     0.973     0.964

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    66/299     12.1G      1.51     0.481         0      1.99         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.84it/s]
                 all       563       597     0.986     0.955     0.979      0.97

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    67/299     12.1G      1.52     0.503         0      2.02         4       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.83it/s]
                 all       563       597     0.983     0.949     0.972     0.965

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    68/299     12.1G      1.59     0.487         0      2.08         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.83it/s]
                 all       563       597     0.959     0.953     0.962     0.956

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    69/299     12.1G      1.39     0.444         0      1.84         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.82it/s]
                 all       563       597     0.915     0.936     0.947     0.925

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    70/299     12.1G      1.52     0.474         0         2         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597     0.948     0.953     0.959      0.95

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    71/299     12.1G      1.44     0.451         0      1.89         4       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597     0.974      0.94     0.957     0.956

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    72/299     12.1G      1.46     0.478         0      1.94         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.82it/s]
                 all       563       597     0.988     0.941     0.969     0.964

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    73/299     12.1G      1.55       0.5         0      2.05         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.81it/s]
                 all       563       597      0.98     0.951     0.971     0.965

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    74/299     12.1G      1.71     0.475         0      2.19         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.83it/s]
                 all       563       597     0.983      0.95     0.971     0.966

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    75/299     12.1G      1.44     0.456         0       1.9         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.85it/s]
                 all       563       597     0.978     0.961     0.975      0.97

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    76/299     12.1G      1.51     0.454         0      1.97         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.84it/s]
                 all       563       597     0.977     0.951     0.967     0.964

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    77/299     12.1G      1.56     0.479         0      2.04         4       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.86it/s]
                 all       563       597     0.981     0.953      0.97     0.967

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    78/299     12.1G      1.58     0.478         0      2.06         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.83it/s]
                 all       563       597     0.973     0.961     0.979     0.967

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    79/299     12.1G      1.41     0.435         0      1.85         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597     0.968     0.956     0.973     0.962

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    80/299     12.1G      1.49     0.454         0      1.94         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.84it/s]
                 all       563       597     0.986     0.951     0.977     0.968

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    81/299     12.1G      1.43     0.461         0      1.89         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.83it/s]
                 all       563       597     0.968      0.96     0.978     0.964

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    82/299     12.1G      1.32     0.452         0      1.77         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.85it/s]
                 all       563       597      0.98     0.964     0.979     0.972

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    83/299     12.1G      1.46     0.429         0      1.89         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.84it/s]
                 all       563       597     0.975     0.961     0.973     0.968

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    84/299     12.1G      1.35     0.439         0      1.79         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.83it/s]
                 all       563       597     0.976     0.956     0.976     0.966

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    85/299     12.1G      1.29     0.456         0      1.74         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.84it/s]
                 all       563       597     0.968     0.958     0.967     0.963

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    86/299     12.1G      1.42     0.429         0      1.85         4       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.83it/s]
                 all       563       597     0.988     0.957     0.971     0.972

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    87/299     12.1G      1.29     0.429         0      1.72         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.86it/s]
                 all       563       597     0.984     0.955     0.972     0.969

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    88/299     12.1G      1.51     0.442         0      1.95         8       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.84it/s]
                 all       563       597     0.986     0.953     0.979     0.969

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    89/299     12.1G       1.4     0.416         0      1.81         4       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.85it/s]
                 all       563       597     0.981     0.957     0.971     0.969

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    90/299     12.1G      1.44      0.45         0      1.89         4       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.84it/s]
                 all       563       597      0.99     0.958     0.975     0.974

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    91/299     12.1G      1.41     0.433         0      1.84         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.83it/s]
                 all       563       597     0.981     0.957     0.971     0.969

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    92/299     12.1G      1.37     0.427         0       1.8         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.83it/s]
                 all       563       597     0.985      0.96     0.972     0.972

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    93/299     12.1G      1.22     0.419         0      1.64         8       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.82it/s]
                 all       563       597     0.983     0.958     0.972      0.97

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    94/299     12.1G      1.35     0.414         0      1.76         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.83it/s]
                 all       563       597     0.985     0.959     0.979     0.972

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    95/299     12.1G      1.41      0.43         0      1.84         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.82it/s]
                 all       563       597     0.986     0.955     0.973      0.97

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    96/299     12.1G      1.41     0.414         0      1.83         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.81it/s]
                 all       563       597     0.978     0.956     0.968     0.967

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    97/299     12.1G      1.36      0.42         0      1.78         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.83it/s]
                 all       563       597     0.984     0.956      0.98      0.97

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    98/299     12.1G      1.44     0.426         0      1.87         8       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.85it/s]
                 all       563       597     0.983     0.961     0.983     0.972

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    99/299     12.1G      1.39     0.436         0      1.82         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.84it/s]
                 all       563       597      0.98     0.958     0.976     0.969

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   100/299     12.1G      1.32     0.432         0      1.75         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.83it/s]
                 all       563       597     0.976     0.962     0.973     0.969

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   101/299     12.1G      1.37     0.428         0       1.8         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.82it/s]
                 all       563       597      0.98     0.966     0.974     0.973

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   102/299     12.1G      1.29     0.419         0      1.71         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.83it/s]
                 all       563       597     0.985     0.966     0.973     0.975

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   103/299     12.1G      1.39     0.414         0      1.81         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.81it/s]
                 all       563       597     0.983     0.953     0.972     0.968

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   104/299     12.1G      1.34     0.442         0      1.78         8       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.85it/s]
                 all       563       597      0.99     0.948     0.971     0.968

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   105/299     12.1G      1.28     0.414         0      1.69         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.86it/s]
                 all       563       597     0.989     0.958      0.97     0.973

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   106/299     12.1G      1.23       0.4         0      1.63         4       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.82it/s]
                 all       563       597     0.983     0.962     0.976     0.972

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   107/299     12.1G      1.36     0.403         0      1.76         4       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.85it/s]
                 all       563       597     0.985     0.959     0.974     0.972

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   108/299     12.1G      1.23     0.397         0      1.62         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597     0.979     0.956     0.977     0.968

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   109/299     12.1G      1.24     0.399         0      1.64         4       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.83it/s]
                 all       563       597     0.988     0.959     0.979     0.973

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   110/299     12.1G      1.24     0.397         0      1.64         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.85it/s]
                 all       563       597     0.983     0.943     0.968     0.963

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   111/299     12.1G      1.21     0.396         0      1.61         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.83it/s]
                 all       563       597     0.987     0.961     0.981     0.974

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   112/299     12.1G       1.3     0.387         0      1.68         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.81it/s]
                 all       563       597      0.99     0.961      0.98     0.975

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   113/299     12.1G      1.25     0.389         0      1.64         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.85it/s]
                 all       563       597     0.986     0.968     0.981     0.977

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   114/299     12.1G      1.19     0.374         0      1.56         5       512: 100% 57/57 [00:37<00:00,  1.53it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.84it/s]
                 all       563       597     0.989     0.965      0.98     0.977

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   115/299     12.1G      1.23     0.397         0      1.63         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.81it/s]
                 all       563       597      0.99     0.958     0.976     0.974

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   116/299     12.1G      1.19     0.395         0      1.59         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.83it/s]
                 all       563       597      0.99     0.964     0.982     0.977

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   117/299     12.1G      1.13     0.381         0      1.51         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.84it/s]
                 all       563       597     0.988     0.964      0.98     0.976

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   118/299     12.1G      1.17     0.414         0      1.59         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.84it/s]
                 all       563       597     0.982     0.955     0.979     0.968

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   119/299     12.1G      1.24     0.401         0      1.64         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.85it/s]
                 all       563       597     0.983     0.959     0.977     0.971

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   120/299     12.1G       1.2     0.388         0      1.59         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.85it/s]
                 all       563       597     0.982     0.953     0.977     0.968

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   121/299     12.1G      1.21     0.376         0      1.59         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.84it/s]
                 all       563       597     0.985     0.958     0.976     0.971

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   122/299     12.1G      1.24     0.389         0      1.63         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.84it/s]
                 all       563       597     0.988     0.959     0.978     0.973

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   123/299     12.1G      1.14     0.368         0       1.5         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.87it/s]
                 all       563       597     0.993     0.969     0.983     0.981

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   124/299     12.1G      1.11     0.402         0      1.52         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.85it/s]
                 all       563       597     0.993     0.968      0.98      0.98

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   125/299     12.1G      1.16     0.359         0      1.52         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597     0.991     0.956     0.973     0.973

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   126/299     12.1G      1.17     0.367         0      1.54         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597     0.988     0.958     0.969     0.973

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   127/299     12.1G       1.1     0.374         0      1.47         4       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.77it/s]
                 all       563       597     0.985     0.961     0.971     0.973

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   128/299     12.1G      1.05      0.37         0      1.42         9       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.72it/s]
                 all       563       597     0.985     0.959     0.976     0.972

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   129/299     12.1G      1.13     0.362         0      1.49         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.75it/s]
                 all       563       597      0.99     0.956     0.977     0.972

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   130/299     12.1G      1.12     0.361         0      1.48         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.83it/s]
                 all       563       597     0.989     0.958     0.979     0.973

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   131/299     12.1G      1.25     0.385         0      1.63         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.82it/s]
                 all       563       597      0.99      0.96      0.98     0.975

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   132/299     12.1G      1.13     0.378         0      1.51         4       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597     0.985     0.959     0.978     0.971

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   133/299     12.1G      1.24     0.398         0      1.63         4       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.78it/s]
                 all       563       597     0.983     0.956      0.97     0.969

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   134/299     12.1G      1.29     0.411         0       1.7         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.78it/s]
                 all       563       597     0.955     0.958     0.973     0.957

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   135/299     12.1G       1.1     0.403         0      1.51         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.77it/s]
                 all       563       597     0.969     0.965     0.975     0.967

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   136/299     12.1G      1.08     0.398         0      1.48         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.71it/s]
                 all       563       597     0.977     0.968     0.982     0.972

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   137/299     12.1G      1.06      0.37         0      1.43         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.78it/s]
                 all       563       597     0.986     0.966     0.981     0.976

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   138/299     12.1G      1.11     0.363         0      1.48         9       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.77it/s]
                 all       563       597     0.986     0.958     0.977     0.972

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   139/299     12.1G       1.1     0.372         0      1.47         4       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.79it/s]
                 all       563       597     0.988     0.955     0.975     0.971

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   140/299     12.1G      1.15     0.389         0      1.54         4       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597     0.983     0.953     0.973     0.968

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   141/299     12.1G      1.12     0.347         0      1.46         8       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.81it/s]
                 all       563       597     0.986     0.955     0.977      0.97

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   142/299     12.1G      1.12     0.367         0      1.49         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.76it/s]
                 all       563       597     0.987     0.961     0.981     0.974

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   143/299     12.1G      1.09     0.391         0      1.48         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.79it/s]
                 all       563       597     0.985     0.958     0.976     0.971

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   144/299     12.1G      1.14     0.356         0       1.5         4       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.79it/s]
                 all       563       597     0.988      0.96     0.968     0.974

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   145/299     12.1G      1.08      0.37         0      1.45         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.79it/s]
                 all       563       597     0.986     0.961     0.977     0.973

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   146/299     12.1G      1.07     0.357         0      1.43         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597     0.981     0.962     0.979     0.972

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   147/299     12.1G      1.08     0.351         0      1.43         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.77it/s]
                 all       563       597     0.988     0.954     0.976     0.971

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   148/299     12.1G      1.08     0.365         0      1.44         8       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.81it/s]
                 all       563       597     0.991      0.96     0.976     0.975

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   149/299     12.1G      1.06     0.365         0      1.43         4       512: 100% 57/57 [00:37<00:00,  1.51it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.79it/s]
                 all       563       597     0.995     0.959     0.977     0.977

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   150/299     12.1G      1.03     0.365         0       1.4         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.79it/s]
                 all       563       597     0.988      0.97     0.975     0.979

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   151/299     12.1G      1.09     0.349         0      1.44         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597     0.983     0.969     0.975     0.976

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   152/299     12.1G      1.11     0.369         0      1.48         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.78it/s]
                 all       563       597     0.974      0.97     0.978     0.972

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   153/299     12.1G      1.06     0.342         0       1.4         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.77it/s]
                 all       563       597     0.979      0.96     0.973      0.97

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   154/299     12.1G      1.02     0.343         0      1.36         9       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.76it/s]
                 all       563       597      0.98     0.965     0.982     0.972

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   155/299     12.1G      1.01     0.327         0      1.34         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.77it/s]
                 all       563       597      0.98     0.968     0.981     0.974

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   156/299     12.1G      1.13     0.337         0      1.47         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.79it/s]
                 all       563       597     0.976     0.968     0.977     0.972

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   157/299     12.1G      1.16     0.343         0      1.51         9       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.78it/s]
                 all       563       597     0.985     0.961      0.98     0.973

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   158/299     12.1G      1.07     0.342         0      1.41         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.81it/s]
                 all       563       597      0.98     0.961     0.979     0.971

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   159/299     12.1G     0.941     0.355         0       1.3         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.77it/s]
                 all       563       597     0.983      0.96     0.973     0.972

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   160/299     12.1G     0.988     0.326         0      1.31         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597     0.988     0.963     0.977     0.975

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   161/299     12.1G      1.03     0.342         0      1.38         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.79it/s]
                 all       563       597     0.995     0.958     0.973     0.976

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   162/299     12.1G      0.96     0.332         0      1.29         9       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.78it/s]
                 all       563       597     0.993     0.959     0.979     0.976

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   163/299     12.1G      1.08     0.363         0      1.45         4       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597     0.993     0.961     0.979     0.977

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   164/299     12.1G       1.1     0.307         0      1.41         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.77it/s]
                 all       563       597     0.985     0.965     0.972     0.975

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   165/299     12.1G      1.03     0.328         0      1.36         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.78it/s]
                 all       563       597      0.98     0.963      0.97     0.972

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   166/299     12.1G      1.02      0.31         0      1.33         8       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.77it/s]
                 all       563       597     0.982     0.968     0.977     0.975

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   167/299     12.1G      1.03     0.312         0      1.34         4       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.81it/s]
                 all       563       597     0.977     0.963     0.969      0.97

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   168/299     12.1G      1.02     0.339         0      1.36         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597     0.979     0.968     0.977     0.974

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   169/299     12.1G     0.963     0.322         0      1.29         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.82it/s]
                 all       563       597     0.985     0.966     0.979     0.976

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   170/299     12.1G      0.95     0.318         0      1.27         8       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.79it/s]
                 all       563       597     0.986     0.963      0.98     0.975

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   171/299     12.1G     0.921     0.317         0      1.24         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.81it/s]
                 all       563       597     0.981     0.966      0.98     0.973

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   172/299     12.1G      0.94     0.306         0      1.25         9       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597      0.98     0.965     0.979     0.972

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   173/299     12.1G      1.01       0.3         0      1.31         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597     0.983     0.966     0.981     0.975

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   174/299     12.1G     0.969     0.331         0       1.3         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.79it/s]
                 all       563       597     0.983      0.97     0.981     0.977

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   175/299     12.1G     0.914      0.31         0      1.22         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597      0.98     0.973     0.977     0.977

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   176/299     12.1G      1.06     0.343         0       1.4        11       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.79it/s]
                 all       563       597     0.985      0.97     0.979     0.977

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   177/299     12.1G     0.924     0.301         0      1.22         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.82it/s]
                 all       563       597     0.984     0.968     0.981     0.976

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   178/299     12.1G     0.943     0.325         0      1.27         4       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597     0.986     0.973     0.983      0.98

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   179/299     12.1G     0.873     0.304         0      1.18        10       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.81it/s]
                 all       563       597      0.98     0.974     0.979     0.977

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   180/299     12.1G     0.881     0.308         0      1.19         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.82it/s]
                 all       563       597     0.984     0.972     0.979     0.978

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   181/299     12.1G     0.849     0.311         0      1.16         9       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597      0.99     0.972     0.981     0.981

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   182/299     12.1G     0.855     0.299         0      1.15         6       512: 100% 57/57 [00:37<00:00,  1.51it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597     0.988     0.965      0.98     0.976

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   183/299     12.1G     0.913     0.318         0      1.23         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.77it/s]
                 all       563       597     0.988      0.97      0.98     0.979

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   184/299     12.1G     0.883     0.287         0      1.17         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.76it/s]
                 all       563       597     0.985      0.97     0.981     0.978

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   185/299     12.1G     0.868     0.308         0      1.18         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.74it/s]
                 all       563       597     0.984     0.972     0.983     0.978

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   186/299     12.1G     0.885     0.279         0      1.16         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.75it/s]
                 all       563       597     0.986     0.977     0.983     0.981

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   187/299     12.1G     0.803     0.305         0      1.11         4       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.79it/s]
                 all       563       597     0.985     0.973      0.98     0.979

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   188/299     12.1G      0.83      0.29         0      1.12         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.79it/s]
                 all       563       597     0.988     0.975     0.982     0.981

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   189/299     12.1G     0.911     0.289         0       1.2         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.77it/s]
                 all       563       597     0.986     0.972     0.981     0.979

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   190/299     12.1G     0.829     0.275         0       1.1         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.78it/s]
                 all       563       597     0.995     0.972     0.982     0.983

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   191/299     12.1G     0.756     0.295         0      1.05         9       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.78it/s]
                 all       563       597     0.991     0.973     0.982     0.982

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   192/299     12.1G     0.888     0.275         0      1.16         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.79it/s]
                 all       563       597     0.993      0.97     0.979     0.981

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   193/299     12.1G     0.867     0.278         0      1.15         4       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.78it/s]
                 all       563       597     0.988     0.969     0.979     0.978

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   194/299     12.1G     0.865     0.269         0      1.13         3       512: 100% 57/57 [00:37<00:00,  1.51it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597     0.988     0.973      0.98      0.98

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   195/299     12.1G     0.895     0.266         0      1.16         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597      0.99     0.972     0.979     0.981

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   196/299     12.1G     0.869       0.3         0      1.17         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.79it/s]
                 all       563       597     0.988     0.969     0.979     0.978

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   197/299     12.1G     0.787     0.278         0      1.07         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597     0.986     0.969      0.98     0.977

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   198/299     12.1G     0.816     0.271         0      1.09         8       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.77it/s]
                 all       563       597     0.987     0.968      0.98     0.978

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   199/299     12.1G     0.811     0.286         0       1.1         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597     0.986     0.966      0.98     0.976

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   200/299     12.1G     0.799     0.273         0      1.07         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.78it/s]
                 all       563       597     0.983     0.966     0.982     0.974

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   201/299     12.1G     0.811     0.275         0      1.09         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.76it/s]
                 all       563       597     0.988     0.963     0.981     0.975

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   202/299     12.1G     0.787     0.277         0      1.06        10       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.78it/s]
                 all       563       597     0.987     0.968     0.982     0.977

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   203/299     12.1G     0.838     0.269         0      1.11         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.76it/s]
                 all       563       597     0.988     0.972     0.982      0.98

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   204/299     12.1G     0.738     0.277         0      1.01         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.81it/s]
                 all       563       597     0.989     0.972     0.982      0.98

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   205/299     12.1G     0.768     0.267         0      1.03         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597     0.992     0.968     0.981      0.98

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   206/299     12.1G     0.899     0.271         0      1.17         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597     0.991      0.97     0.982     0.981

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   207/299     12.1G     0.802     0.291         0      1.09         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.79it/s]
                 all       563       597     0.985     0.977     0.982     0.981

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   208/299     12.1G      0.78     0.261         0      1.04         4       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.79it/s]
                 all       563       597     0.991     0.973     0.981     0.982

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   209/299     12.1G     0.837     0.278         0      1.12         8       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.81it/s]
                 all       563       597     0.988     0.974     0.982     0.981

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   210/299     12.1G     0.877     0.255         0      1.13         4       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.81it/s]
                 all       563       597      0.99     0.972     0.978     0.981

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   211/299     12.1G     0.785      0.25         0      1.03         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.79it/s]
                 all       563       597     0.986     0.971     0.973     0.978

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   212/299     12.1G     0.751     0.272         0      1.02         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.81it/s]
                 all       563       597     0.985     0.972     0.973     0.978

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   213/299     12.1G     0.715     0.265         0      0.98         4       512: 100% 57/57 [00:37<00:00,  1.51it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597     0.993     0.974     0.974     0.983

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   214/299     12.1G     0.846     0.253         0       1.1         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.77it/s]
                 all       563       597     0.986     0.972     0.978     0.979

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   215/299     12.1G     0.752     0.256         0      1.01         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597     0.988     0.972     0.979      0.98

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   216/299     12.1G     0.743     0.253         0     0.996         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.79it/s]
                 all       563       597     0.992     0.972      0.98     0.981

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   217/299     12.1G     0.676     0.257         0     0.933         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597     0.993     0.975     0.981     0.984

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   218/299     12.1G     0.764      0.28         0      1.04         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.79it/s]
                 all       563       597      0.99     0.979     0.982     0.985

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   219/299     12.1G     0.644     0.238         0     0.882         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.79it/s]
                 all       563       597     0.988     0.976     0.983     0.982

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   220/299     12.1G     0.753      0.26         0      1.01         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597      0.99     0.974     0.983     0.982

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   221/299     12.1G      0.73     0.265         0     0.995         4       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597      0.99     0.973     0.983     0.981

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   222/299     12.1G     0.687      0.26         0     0.947         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.79it/s]
                 all       563       597     0.989     0.975     0.983     0.982

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   223/299     12.1G     0.699     0.262         0      0.96        11       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597     0.989     0.973      0.98     0.981

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   224/299     12.1G     0.751     0.241         0     0.992         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.78it/s]
                 all       563       597     0.993     0.974     0.981     0.984

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   225/299     12.1G     0.765     0.264         0      1.03         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.81it/s]
                 all       563       597     0.992     0.975     0.982     0.983

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   226/299     12.1G     0.741     0.244         0     0.985         4       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597     0.992     0.975     0.983     0.983

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   227/299     12.1G     0.717     0.248         0     0.966         3       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.78it/s]
                 all       563       597     0.993     0.972      0.98     0.982

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   228/299     12.1G      0.73     0.267         0     0.997         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.79it/s]
                 all       563       597     0.991     0.969     0.979      0.98

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   229/299     12.1G     0.752     0.233         0     0.986        12       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597      0.99      0.97      0.98      0.98

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   230/299     12.1G     0.686     0.251         0     0.937         4       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.82it/s]
                 all       563       597      0.99     0.972     0.982     0.981

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   231/299     12.1G     0.686      0.25         0     0.935         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597     0.991     0.975     0.982     0.983

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   232/299     12.1G     0.709     0.254         0     0.964         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.78it/s]
                 all       563       597     0.991     0.974     0.979     0.983

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   233/299     12.1G     0.729     0.275         0         1         8       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.81it/s]
                 all       563       597     0.991     0.973     0.979     0.982

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   234/299     12.1G     0.672     0.228         0       0.9         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.81it/s]
                 all       563       597      0.99     0.973      0.98     0.982

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   235/299     12.1G     0.739     0.238         0     0.977         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597     0.992     0.975     0.982     0.984

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   236/299     12.1G       0.7     0.244         0     0.944         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.81it/s]
                 all       563       597      0.99     0.972     0.982     0.981

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   237/299     12.1G     0.692     0.253         0     0.945         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.81it/s]
                 all       563       597      0.99     0.975     0.983     0.983

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   238/299     12.1G     0.743     0.222         0     0.965         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597     0.989     0.975     0.983     0.982

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   239/299     12.1G     0.609     0.233         0     0.842         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.73it/s]
                 all       563       597     0.987     0.973      0.98      0.98

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   240/299     12.1G     0.647     0.236         0     0.883         4       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.79it/s]
                 all       563       597     0.988     0.976     0.984     0.982

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   241/299     12.1G     0.665      0.24         0     0.904         4       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597      0.99     0.976     0.984     0.983

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   242/299     12.1G     0.655     0.228         0     0.882         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.82it/s]
                 all       563       597     0.988     0.976     0.984     0.982

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   243/299     12.1G     0.672     0.235         0     0.907         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.82it/s]
                 all       563       597     0.991     0.977     0.983     0.984

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   244/299     12.1G     0.573     0.242         0     0.815         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.82it/s]
                 all       563       597     0.992     0.978     0.983     0.984

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   245/299     12.1G     0.656     0.227         0     0.883         4       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.81it/s]
                 all       563       597     0.991     0.977     0.984     0.984

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   246/299     12.1G     0.696     0.223         0     0.919         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597     0.992     0.977     0.984     0.984

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   247/299     12.1G     0.625     0.227         0     0.852         4       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.82it/s]
                 all       563       597     0.992     0.977     0.983     0.984

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   248/299     12.1G     0.626      0.23         0     0.855         4       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597     0.991     0.976     0.984     0.984

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   249/299     12.1G     0.615     0.227         0     0.842         4       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.83it/s]
                 all       563       597     0.992     0.977     0.983     0.984

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   250/299     12.1G     0.603     0.252         0     0.855         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.79it/s]
                 all       563       597     0.992     0.977     0.983     0.984

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   251/299     12.1G     0.576     0.234         0     0.811         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597      0.99     0.975     0.983     0.983

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   252/299     12.1G     0.637      0.24         0     0.877         4       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.81it/s]
                 all       563       597      0.99     0.974     0.983     0.982

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   253/299     12.1G     0.692     0.225         0     0.917         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.83it/s]
                 all       563       597     0.992     0.978     0.984     0.985

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   254/299     12.1G     0.615     0.216         0     0.831         4       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.81it/s]
                 all       563       597      0.99     0.974     0.983     0.982

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   255/299     12.1G     0.633     0.225         0     0.858         4       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.82it/s]
                 all       563       597      0.99     0.976     0.982     0.983

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   256/299     12.1G     0.732     0.229         0     0.961        11       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.82it/s]
                 all       563       597     0.989     0.975     0.982     0.982

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   257/299     12.1G     0.606     0.235         0     0.841         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.81it/s]
                 all       563       597      0.99     0.974     0.982     0.982

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   258/299     12.1G     0.615     0.224         0      0.84         4       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.82it/s]
                 all       563       597     0.991     0.971     0.982     0.981

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   259/299     12.1G      0.57     0.221         0     0.791         8       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597     0.991     0.971     0.982     0.981

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   260/299     12.1G     0.547     0.218         0     0.766         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.77it/s]
                 all       563       597     0.991     0.974     0.983     0.983

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   261/299     12.1G     0.543     0.236         0      0.78         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597     0.991     0.974     0.983     0.983

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   262/299     12.1G     0.561     0.215         0     0.776         8       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.79it/s]
                 all       563       597     0.991     0.972     0.983     0.981

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   263/299     12.1G     0.589     0.228         0     0.817         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.79it/s]
                 all       563       597     0.993     0.974     0.983     0.984

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   264/299     12.1G     0.622     0.212         0     0.834         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.77it/s]
                 all       563       597     0.992     0.973     0.982     0.982

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   265/299     12.1G      0.54     0.214         0     0.753         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.78it/s]
                 all       563       597     0.993     0.975     0.981     0.984

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   266/299     12.1G     0.548     0.221         0     0.769         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597     0.994     0.975     0.983     0.984

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   267/299     12.1G     0.586     0.231         0     0.817         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597     0.993     0.975     0.983     0.984

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   268/299     12.1G     0.579     0.207         0     0.785         9       512: 100% 57/57 [00:37<00:00,  1.51it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597     0.993     0.977     0.983     0.984

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   269/299     12.1G     0.624     0.196         0      0.82         4       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597     0.992     0.977     0.982     0.984

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   270/299     12.1G      0.61     0.218         0     0.828         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.78it/s]
                 all       563       597     0.992     0.977     0.982     0.984

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   271/299     12.1G     0.613     0.222         0     0.835        11       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.79it/s]
                 all       563       597     0.992     0.977     0.983     0.984

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   272/299     12.1G     0.611      0.21         0     0.821         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597     0.992     0.977     0.983     0.984

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   273/299     12.1G     0.612      0.22         0     0.832         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.78it/s]
                 all       563       597     0.992     0.977     0.984     0.984

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   274/299     12.1G     0.566     0.232         0     0.798         8       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597     0.991     0.977     0.983     0.984

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   275/299     12.1G     0.585     0.205         0      0.79         8       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597      0.99     0.975     0.983     0.982

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   276/299     12.1G     0.557     0.201         0     0.758         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.81it/s]
                 all       563       597      0.99     0.974     0.982     0.982

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   277/299     12.1G     0.574     0.214         0     0.788         3       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.81it/s]
                 all       563       597     0.989     0.973     0.982     0.981

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   278/299     12.1G      0.63     0.216         0     0.846         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597     0.988     0.973     0.982     0.981

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   279/299     12.1G     0.586     0.217         0     0.802         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.81it/s]
                 all       563       597     0.988     0.976     0.982     0.982

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   280/299     12.1G     0.575     0.218         0     0.794         4       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.83it/s]
                 all       563       597     0.988     0.977     0.982     0.982

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   281/299     12.1G     0.624     0.229         0     0.854         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.81it/s]
                 all       563       597     0.989     0.977     0.982     0.983

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   282/299     12.1G     0.541     0.215         0     0.756         3       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.85it/s]
                 all       563       597     0.989     0.977     0.982     0.983

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   283/299     12.1G     0.506     0.217         0     0.724         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.79it/s]
                 all       563       597     0.989     0.977     0.983     0.983

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   284/299     12.1G      0.55      0.21         0     0.761         8       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.81it/s]
                 all       563       597      0.99     0.976     0.982     0.983

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   285/299     12.1G      0.58     0.224         0     0.805         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597     0.992     0.978     0.983     0.985

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   286/299     12.1G     0.552     0.184         0     0.736         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.79it/s]
                 all       563       597     0.991     0.977     0.983     0.984

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   287/299     12.1G     0.597      0.21         0     0.807         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.81it/s]
                 all       563       597     0.991     0.975     0.983     0.983

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   288/299     12.1G     0.561     0.206         0     0.767         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597     0.993     0.977     0.983     0.985

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   289/299     12.1G     0.546     0.216         0     0.762         4       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.82it/s]
                 all       563       597     0.991     0.975     0.983     0.983

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   290/299     12.1G     0.591     0.213         0     0.804         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.82it/s]
                 all       563       597     0.991     0.975     0.983     0.983

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   291/299     12.1G     0.517     0.212         0     0.729         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.81it/s]
                 all       563       597     0.991     0.975     0.983     0.983

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   292/299     12.1G     0.555     0.191         0     0.746         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.81it/s]
                 all       563       597     0.993     0.975     0.983     0.984

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   293/299     12.1G     0.529     0.201         0     0.729         4       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.80it/s]
                 all       563       597     0.993     0.975     0.983     0.984

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   294/299     12.1G     0.537     0.198         0     0.735         5       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.81it/s]
                 all       563       597     0.992     0.973     0.982     0.982

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   295/299     12.1G     0.549     0.199         0     0.748         3       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.79it/s]
                 all       563       597     0.992     0.973     0.982     0.983

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   296/299     12.1G     0.644     0.196         0     0.841         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.79it/s]
                 all       563       597     0.993     0.974     0.983     0.984

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   297/299     12.1G     0.651     0.218         0     0.868         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.79it/s]
                 all       563       597     0.992     0.973     0.982     0.982

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   298/299     12.1G     0.594     0.216         0      0.81         7       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.81it/s]
                 all       563       597     0.994     0.975     0.983     0.984

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   299/299     12.1G     0.585     0.217         0     0.802         6       512: 100% 57/57 [00:37<00:00,  1.52it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 57/57 [00:09<00:00,  5.77it/s]
                 all       563       597     0.995     0.976     0.983     0.985
300 epochs completed in 3.961 hours.
