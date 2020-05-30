## EVA Assignment 12
----------------------
## Name : Abhijit Mali
----------------------
## Notes 
---------------------------------------------------------------------------------------------------------------------------
1. Achieved Max Accuracy of 54.42%. Crossed 50% accuracy 4 times. Used onecycle policy with annealing i.e. traingular2 mode

![triangleplot](https://github.com/csharpshooter/EVA/blob/master/A12/A12-A/images/TrianglePlot.png)

2. Read images from train and val folder folders after extracting and did 70-30 train-val split in memory. Parsed train and test data by returning image path and labels from dataset files. Stored image paths and loaded image from disk in tinyimagenet dataset in __getitem__ method. Loaded class labels from words.txt and loaded class id's from wnids.txt

![Imgfromdataset](https://github.com/csharpshooter/EVA/blob/master/A12/A12-A/images/imagesfromdataset.png)

3. Ran model for 20 epochs for lr range test. LR range tested from 0.01 to 0.1. Found optimal Lr between 0.04935 to 0.0038.
Max LR = 0.068 ,Min LR = Max LR / 13 = 0.00523

![lrrangefinder](https://github.com/csharpshooter/EVA/blob/master/A12/A12-A/images/lrrangetestgraph.png)

4. Max train accuracy =  78.63, max test accuracy = 54.42. Tried different training method this time. Cloned git repo to colab and then trained from colab

5. Used following pytorch transforms for augmentation:
  *  RandomRotate(20),
  *  RandomHorizontalFlip,
  *  RandomCrop(size=(64, 64), padding=4),
  *  RandomErasing(scale=(0.10, 0.10), ratio=(1, 1)), 

 
 6. Added Tensorboard visualization

 ![ModelGraph](https://github.com/csharpshooter/EVA/blob/master/A12/A12-A/images/ModelGraphTensorBoard.png)
 ![Graphs](https://github.com/csharpshooter/EVA/blob/master/A12/A12-A/images/TensorBoardGraphs.png)
 ![Dist](https://github.com/csharpshooter/EVA/blob/master/A12/A12-A/images/TensorBoardDistribution.png)
 ![Hist](https://github.com/csharpshooter/EVA/blob/master/A12/A12-A/images/TensorBoardHistogram.png)
    
---------------------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------------------------------
## Project Structure
--------------------

![ProjectStructure](https://github.com/csharpshooter/EVA/blob/master/A11/images/projectstructure.png)

---------------------------------------------------------------------------------------------------------------------------
## Test and Train, Loss and Accuracy Graphs

![Graphs](https://github.com/csharpshooter/EVA/blob/master/A12/A12-A/images/traintestgraphs.png)

---------------------------------------------------------------------------------------------------------------------------
## Model Summary

![ModelSummary](https://github.com/csharpshooter/EVA/blob/master/A12/A12-A/images/modelsummary.png)

---------------------------------------------------------------------------------------------------------------------------
## Logs

  0%|          | 0/151 [00:00<?, ?it/s]EPOCH: 0
Learning rate = 0.0038  for epoch:  0
Loss=4.822839260101318 Batch_id=150 Accuracy=4.25: 100%|██████████| 151/151 [02:12<00:00,  1.14it/s]
  0%|          | 0/151 [00:00<?, ?it/s]
Test set: Average loss: 0.0092, Accuracy: 2015/33000 (6.11%)

Validation accuracy increased (0.000000 --> 6.106061).  Saving model ...
EPOCH: 1
Learning rate = 0.012910000000000008  for epoch:  1
Loss=3.924772024154663 Batch_id=150 Accuracy=9.76: 100%|██████████| 151/151 [02:12<00:00,  1.14it/s]
  0%|          | 0/151 [00:00<?, ?it/s]
Test set: Average loss: 0.0090, Accuracy: 2534/33000 (7.68%)

Validation accuracy increased (6.106061 --> 7.678788).  Saving model ...
EPOCH: 2
Learning rate = 0.02202000000000002  for epoch:  2
Loss=3.6599209308624268 Batch_id=150 Accuracy=15.18: 100%|██████████| 151/151 [02:12<00:00,  1.14it/s]
  0%|          | 0/151 [00:00<?, ?it/s]
Test set: Average loss: 0.0076, Accuracy: 5167/33000 (15.66%)

Validation accuracy increased (7.678788 --> 15.657576).  Saving model ...
EPOCH: 3
Learning rate = 0.031130000000000026  for epoch:  3
Loss=3.2844152450561523 Batch_id=150 Accuracy=19.67: 100%|██████████| 151/151 [02:12<00:00,  1.14it/s]
  0%|          | 0/151 [00:00<?, ?it/s]
Test set: Average loss: 0.0074, Accuracy: 6120/33000 (18.55%)

Validation accuracy increased (15.657576 --> 18.545455).  Saving model ...
EPOCH: 4
Learning rate = 0.04023999999999998  for epoch:  4
Loss=3.5558042526245117 Batch_id=150 Accuracy=23.29: 100%|██████████| 151/151 [02:12<00:00,  1.14it/s]
  0%|          | 0/151 [00:00<?, ?it/s]
Test set: Average loss: 0.0081, Accuracy: 4817/33000 (14.60%)

EPOCH: 5
Learning rate = 0.04934999999999999  for epoch:  5
Loss=3.0426547527313232 Batch_id=150 Accuracy=26.39: 100%|██████████| 151/151 [02:12<00:00,  1.14it/s]
  0%|          | 0/151 [00:00<?, ?it/s]
Test set: Average loss: 0.0093, Accuracy: 3386/33000 (10.26%)

EPOCH: 6
Learning rate = 0.047072499999999996  for epoch:  6
Loss=3.021240472793579 Batch_id=150 Accuracy=29.81: 100%|██████████| 151/151 [02:12<00:00,  1.14it/s]
  0%|          | 0/151 [00:00<?, ?it/s]
Test set: Average loss: 0.0078, Accuracy: 5300/33000 (16.06%)

EPOCH: 7
Learning rate = 0.044794999999999995  for epoch:  7
Loss=2.918788433074951 Batch_id=150 Accuracy=32.50: 100%|██████████| 151/151 [02:12<00:00,  1.14it/s]
  0%|          | 0/151 [00:00<?, ?it/s]
Test set: Average loss: 0.0063, Accuracy: 8408/33000 (25.48%)

Validation accuracy increased (18.545455 --> 25.478788).  Saving model ...
EPOCH: 8
Learning rate = 0.04251749999999999  for epoch:  8
Loss=2.9398016929626465 Batch_id=150 Accuracy=34.88: 100%|██████████| 151/151 [02:12<00:00,  1.14it/s]
  0%|          | 0/151 [00:00<?, ?it/s]
Test set: Average loss: 0.0069, Accuracy: 7589/33000 (23.00%)

EPOCH: 9
Learning rate = 0.040240000000000005  for epoch:  9
Loss=2.6281609535217285 Batch_id=150 Accuracy=36.62: 100%|██████████| 151/151 [02:12<00:00,  1.14it/s]
  0%|          | 0/151 [00:00<?, ?it/s]
Test set: Average loss: 0.0081, Accuracy: 5737/33000 (17.38%)

EPOCH: 10
Learning rate = 0.0379625  for epoch:  10
Loss=2.5468602180480957 Batch_id=150 Accuracy=38.55: 100%|██████████| 151/151 [02:12<00:00,  1.14it/s]
  0%|          | 0/151 [00:00<?, ?it/s]
Test set: Average loss: 0.0056, Accuracy: 10707/33000 (32.45%)

Validation accuracy increased (25.478788 --> 32.445455).  Saving model ...
EPOCH: 11
Learning rate = 0.035685  for epoch:  11
Loss=2.4116506576538086 Batch_id=150 Accuracy=40.12: 100%|██████████| 151/151 [02:12<00:00,  1.14it/s]
  0%|          | 0/151 [00:00<?, ?it/s]
Test set: Average loss: 0.0063, Accuracy: 8900/33000 (26.97%)

EPOCH: 12
Learning rate = 0.0334075  for epoch:  12
Loss=2.459543228149414 Batch_id=150 Accuracy=41.65: 100%|██████████| 151/151 [02:12<00:00,  1.14it/s]
  0%|          | 0/151 [00:00<?, ?it/s]
Test set: Average loss: 0.0059, Accuracy: 10275/33000 (31.14%)

EPOCH: 13
Learning rate = 0.03113  for epoch:  13
Loss=2.344982624053955 Batch_id=150 Accuracy=43.10: 100%|██████████| 151/151 [02:12<00:00,  1.14it/s]
  0%|          | 0/151 [00:00<?, ?it/s]
Test set: Average loss: 0.0058, Accuracy: 10492/33000 (31.79%)

EPOCH: 14
Learning rate = 0.0288525  for epoch:  14
Loss=2.4819231033325195 Batch_id=150 Accuracy=44.56: 100%|██████████| 151/151 [02:12<00:00,  1.14it/s]
  0%|          | 0/151 [00:00<?, ?it/s]
Test set: Average loss: 0.0062, Accuracy: 9656/33000 (29.26%)

EPOCH: 15
Learning rate = 0.026574999999999998  for epoch:  15
Loss=2.27528977394104 Batch_id=150 Accuracy=46.05: 100%|██████████| 151/151 [02:12<00:00,  1.14it/s]
  0%|          | 0/151 [00:00<?, ?it/s]
Test set: Average loss: 0.0066, Accuracy: 8858/33000 (26.84%)

EPOCH: 16
Learning rate = 0.024297499999999993  for epoch:  16
Loss=2.1358072757720947 Batch_id=150 Accuracy=47.39: 100%|██████████| 151/151 [02:12<00:00,  1.14it/s]
  0%|          | 0/151 [00:00<?, ?it/s]
Test set: Average loss: 0.0052, Accuracy: 12389/33000 (37.54%)

Validation accuracy increased (32.445455 --> 37.542424).  Saving model ...
EPOCH: 17
Learning rate = 0.02201999999999999  for epoch:  17
Loss=1.8324198722839355 Batch_id=150 Accuracy=49.02: 100%|██████████| 151/151 [02:12<00:00,  1.14it/s]
  0%|          | 0/151 [00:00<?, ?it/s]
Test set: Average loss: 0.0048, Accuracy: 13521/33000 (40.97%)

Validation accuracy increased (37.542424 --> 40.972727).  Saving model ...
EPOCH: 18
Learning rate = 0.019742500000000003  for epoch:  18
Loss=2.04144024848938 Batch_id=150 Accuracy=50.38: 100%|██████████| 151/151 [02:12<00:00,  1.14it/s]
  0%|          | 0/151 [00:00<?, ?it/s]
Test set: Average loss: 0.0050, Accuracy: 12844/33000 (38.92%)

EPOCH: 19
Learning rate = 0.017465  for epoch:  19
Loss=1.987189531326294 Batch_id=150 Accuracy=52.01: 100%|██████████| 151/151 [02:12<00:00,  1.14it/s]
  0%|          | 0/151 [00:00<?, ?it/s]
Test set: Average loss: 0.0051, Accuracy: 12558/33000 (38.05%)

EPOCH: 20
Learning rate = 0.015187499999999998  for epoch:  20
Loss=1.729616403579712 Batch_id=150 Accuracy=53.78: 100%|██████████| 151/151 [02:12<00:00,  1.14it/s]
  0%|          | 0/151 [00:00<?, ?it/s]
Test set: Average loss: 0.0053, Accuracy: 12474/33000 (37.80%)

EPOCH: 21
Learning rate = 0.012910000000000008  for epoch:  21
Loss=1.7213836908340454 Batch_id=150 Accuracy=55.35: 100%|██████████| 151/151 [02:12<00:00,  1.14it/s]

Test set: Average loss: 0.0046, Accuracy: 14333/33000 (43.43%)

Validation accuracy increased (40.972727 --> 43.433333).  Saving model ...
  0%|          | 0/151 [00:00<?, ?it/s]EPOCH: 22
Learning rate = 0.010632500000000007  for epoch:  22
Loss=1.9374393224716187 Batch_id=150 Accuracy=57.32: 100%|██████████| 151/151 [02:12<00:00,  1.14it/s]
  0%|          | 0/151 [00:00<?, ?it/s]
Test set: Average loss: 0.0043, Accuracy: 15408/33000 (46.69%)

Validation accuracy increased (43.433333 --> 46.690909).  Saving model ...
EPOCH: 23
Learning rate = 0.008355000000000005  for epoch:  23
Loss=1.7135740518569946 Batch_id=150 Accuracy=59.68: 100%|██████████| 151/151 [02:12<00:00,  1.14it/s]
  0%|          | 0/151 [00:00<?, ?it/s]
Test set: Average loss: 0.0043, Accuracy: 15621/33000 (47.34%)

Validation accuracy increased (46.690909 --> 47.336364).  Saving model ...
EPOCH: 24
Learning rate = 0.006077500000000002  for epoch:  24
Loss=1.8768616914749146 Batch_id=150 Accuracy=62.10: 100%|██████████| 151/151 [02:12<00:00,  1.14it/s]
  0%|          | 0/151 [00:00<?, ?it/s]
Test set: Average loss: 0.0041, Accuracy: 16095/33000 (48.77%)

Validation accuracy increased (47.336364 --> 48.772727).  Saving model ...
EPOCH: 25
Learning rate = 0.0038  for epoch:  25
Loss=1.5093199014663696 Batch_id=150 Accuracy=65.43: 100%|██████████| 151/151 [02:12<00:00,  1.14it/s]
  0%|          | 0/151 [00:00<?, ?it/s]
Test set: Average loss: 0.0038, Accuracy: 17157/33000 (51.99%)

Validation accuracy increased (48.772727 --> 51.990909).  Saving model ...
EPOCH: 26
Learning rate = 0.008355000000000005  for epoch:  26
Loss=1.4918261766433716 Batch_id=150 Accuracy=61.48: 100%|██████████| 151/151 [02:12<00:00,  1.14it/s]
  0%|          | 0/151 [00:00<?, ?it/s]
Test set: Average loss: 0.0044, Accuracy: 15049/33000 (45.60%)

EPOCH: 27
Learning rate = 0.012910000000000008  for epoch:  27
Loss=1.8018401861190796 Batch_id=150 Accuracy=57.68: 100%|██████████| 151/151 [02:12<00:00,  1.14it/s]
  0%|          | 0/151 [00:00<?, ?it/s]
Test set: Average loss: 0.0050, Accuracy: 13064/33000 (39.59%)

EPOCH: 28
Learning rate = 0.017465000000000012  for epoch:  28
Loss=1.8762071132659912 Batch_id=150 Accuracy=55.35: 100%|██████████| 151/151 [02:12<00:00,  1.14it/s]
  0%|          | 0/151 [00:00<?, ?it/s]
Test set: Average loss: 0.0052, Accuracy: 12550/33000 (38.03%)

EPOCH: 29
Learning rate = 0.02202000000000002  for epoch:  29
Loss=1.927603006362915 Batch_id=150 Accuracy=54.06: 100%|██████████| 151/151 [02:12<00:00,  1.14it/s]
  0%|          | 0/151 [00:00<?, ?it/s]
Test set: Average loss: 0.0056, Accuracy: 11596/33000 (35.14%)

EPOCH: 30
Learning rate = 0.026574999999999998  for epoch:  30
Loss=1.9492545127868652 Batch_id=150 Accuracy=53.64: 100%|██████████| 151/151 [02:12<00:00,  1.14it/s]
  0%|          | 0/151 [00:00<?, ?it/s]
Test set: Average loss: 0.0058, Accuracy: 11056/33000 (33.50%)

EPOCH: 31
Learning rate = 0.025436249999999997  for epoch:  31
Loss=1.9449553489685059 Batch_id=150 Accuracy=55.03: 100%|██████████| 151/151 [02:12<00:00,  1.14it/s]
  0%|          | 0/151 [00:00<?, ?it/s]
Test set: Average loss: 0.0056, Accuracy: 11876/33000 (35.99%)

EPOCH: 32
Learning rate = 0.024297499999999993  for epoch:  32
Loss=1.9479403495788574 Batch_id=150 Accuracy=56.40: 100%|██████████| 151/151 [02:12<00:00,  1.14it/s]
  0%|          | 0/151 [00:00<?, ?it/s]
Test set: Average loss: 0.0053, Accuracy: 12262/33000 (37.16%)

EPOCH: 33
Learning rate = 0.023158749999999992  for epoch:  33
Loss=1.9007365703582764 Batch_id=150 Accuracy=57.34: 100%|██████████| 151/151 [02:12<00:00,  1.14it/s]
  0%|          | 0/151 [00:00<?, ?it/s]
Test set: Average loss: 0.0052, Accuracy: 12470/33000 (37.79%)

EPOCH: 34
Learning rate = 0.02201999999999999  for epoch:  34
Loss=1.5395017862319946 Batch_id=150 Accuracy=58.29: 100%|██████████| 151/151 [02:12<00:00,  1.14it/s]
  0%|          | 0/151 [00:00<?, ?it/s]
Test set: Average loss: 0.0045, Accuracy: 15061/33000 (45.64%)

EPOCH: 35
Learning rate = 0.020881250000000004  for epoch:  35
Loss=1.4190659523010254 Batch_id=150 Accuracy=59.18: 100%|██████████| 151/151 [02:12<00:00,  1.14it/s]
  0%|          | 0/151 [00:00<?, ?it/s]
Test set: Average loss: 0.0056, Accuracy: 11770/33000 (35.67%)

EPOCH: 36
Learning rate = 0.019742500000000003  for epoch:  36
Loss=1.8576340675354004 Batch_id=150 Accuracy=60.32: 100%|██████████| 151/151 [02:12<00:00,  1.14it/s]
  0%|          | 0/151 [00:00<?, ?it/s]
Test set: Average loss: 0.0052, Accuracy: 12540/33000 (38.00%)

EPOCH: 37
Learning rate = 0.018603750000000002  for epoch:  37
Loss=1.878828763961792 Batch_id=150 Accuracy=61.22: 100%|██████████| 151/151 [02:12<00:00,  1.14it/s]
  0%|          | 0/151 [00:00<?, ?it/s]
Test set: Average loss: 0.0049, Accuracy: 13986/33000 (42.38%)

EPOCH: 38
Learning rate = 0.017465  for epoch:  38
Loss=1.5395987033843994 Batch_id=150 Accuracy=61.93: 100%|██████████| 151/151 [02:12<00:00,  1.14it/s]
  0%|          | 0/151 [00:00<?, ?it/s]
Test set: Average loss: 0.0046, Accuracy: 14767/33000 (44.75%)

EPOCH: 39
Learning rate = 0.01632625  for epoch:  39
Loss=1.623794674873352 Batch_id=150 Accuracy=63.03: 100%|██████████| 151/151 [02:12<00:00,  1.14it/s]
  0%|          | 0/151 [00:00<?, ?it/s]
Test set: Average loss: 0.0044, Accuracy: 15462/33000 (46.85%)

EPOCH: 40
Learning rate = 0.015187499999999998  for epoch:  40
Loss=1.7407375574111938 Batch_id=150 Accuracy=64.09: 100%|██████████| 151/151 [02:12<00:00,  1.14it/s]
  0%|          | 0/151 [00:00<?, ?it/s]
Test set: Average loss: 0.0046, Accuracy: 14883/33000 (45.10%)

EPOCH: 41
Learning rate = 0.01404875000000001  for epoch:  41
Loss=1.3801181316375732 Batch_id=150 Accuracy=65.10: 100%|██████████| 151/151 [02:12<00:00,  1.14it/s]
  0%|          | 0/151 [00:00<?, ?it/s]
Test set: Average loss: 0.0049, Accuracy: 13991/33000 (42.40%)

EPOCH: 42
Learning rate = 0.012910000000000008  for epoch:  42
Loss=1.5492231845855713 Batch_id=150 Accuracy=66.35: 100%|██████████| 151/151 [02:12<00:00,  1.14it/s]
  0%|          | 0/151 [00:00<?, ?it/s]
Test set: Average loss: 0.0044, Accuracy: 15621/33000 (47.34%)

EPOCH: 43
Learning rate = 0.011771250000000007  for epoch:  43
Loss=1.5249220132827759 Batch_id=150 Accuracy=67.71: 100%|██████████| 151/151 [02:12<00:00,  1.14it/s]
  0%|          | 0/151 [00:00<?, ?it/s]
Test set: Average loss: 0.0044, Accuracy: 15649/33000 (47.42%)

EPOCH: 44
Learning rate = 0.010632500000000007  for epoch:  44
Loss=1.3411844968795776 Batch_id=150 Accuracy=69.29: 100%|██████████| 151/151 [02:12<00:00,  1.14it/s]
  0%|          | 0/151 [00:00<?, ?it/s]
Test set: Average loss: 0.0042, Accuracy: 16133/33000 (48.89%)

EPOCH: 45
Learning rate = 0.009493750000000006  for epoch:  45
Loss=1.226919412612915 Batch_id=150 Accuracy=70.64: 100%|██████████| 151/151 [02:12<00:00,  1.14it/s]
  0%|          | 0/151 [00:00<?, ?it/s]
Test set: Average loss: 0.0043, Accuracy: 16064/33000 (48.68%)

EPOCH: 46
Learning rate = 0.008355000000000005  for epoch:  46
Loss=1.2630033493041992 Batch_id=150 Accuracy=72.28: 100%|██████████| 151/151 [02:12<00:00,  1.14it/s]
  0%|          | 0/151 [00:00<?, ?it/s]
Test set: Average loss: 0.0041, Accuracy: 16595/33000 (50.29%)

EPOCH: 47
Learning rate = 0.007216250000000003  for epoch:  47
Loss=1.183948040008545 Batch_id=150 Accuracy=74.41: 100%|██████████| 151/151 [02:12<00:00,  1.14it/s]
  0%|          | 0/151 [00:00<?, ?it/s]
Test set: Average loss: 0.0042, Accuracy: 16456/33000 (49.87%)

EPOCH: 48
Learning rate = 0.006077500000000002  for epoch:  48
Loss=1.0493015050888062 Batch_id=150 Accuracy=76.26: 100%|██████████| 151/151 [02:12<00:00,  1.14it/s]
  0%|          | 0/151 [00:00<?, ?it/s]
Test set: Average loss: 0.0041, Accuracy: 16582/33000 (50.25%)

EPOCH: 49
Learning rate = 0.004938750000000001  for epoch:  49
Loss=1.185178279876709 Batch_id=150 Accuracy=78.63: 100%|██████████| 151/151 [02:12<00:00,  1.14it/s]

Test set: Average loss: 0.0038, Accuracy: 17958/33000 (54.42%)

Validation accuracy increased (51.990909 --> 54.418182).  Saving model ...
Saving final model after training cycle completion
