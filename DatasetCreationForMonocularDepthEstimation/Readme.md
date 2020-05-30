------------------------------------------------------------------------------------------------------------
## Name : Abhijit Mali

----------------------
## Notes 
---------------------------------------------------------------------------------------------------------------------------

### Create this dataset and share a link to GDrive (publicly available to anyone) in this readme file. https://drive.google.com/drive/folders/1NXUBngY_z3M6d9DYs8KQZTXo9b7zIF8u?usp=sharing

![f](https://github.com/csharpshooter/EVA/blob/master/A15-PartA/Images/FolderStructure1.JPG)
![f](https://github.com/csharpshooter/EVA/blob/master/A15-PartA/Images/FolderStructure2.JPG)
![f](https://github.com/csharpshooter/EVA/blob/master/A15-PartA/Images/FolderStructure3.JPG)

### Add your dataset statistics:

### Kinds of images (fg, bg, fg_bg, masks, depth)

fg :- Different Man, Woman, kids, group of person(for background transparency we have taken png images) bg :- We restricted background to library images(for restricting size of image we have taken jpg images) fg_bg :- bg superposed over fg (for restricting size of images we have taken jpg images) masks :- masks extracted from fg images(we have taken grayscale images)(.jpg) depth :- We have extracted depth images from fg_bg using nyu model(for restricing size of images we have taken grayscale images extracted from colormap)(.jpg)

### Total images of each kind
fg :- 200(flip + no flip), bg :- 100, fg_bg :- 392783, masks :- 392469, depth :- 394673 

### Note : We have all generated 400k images of each fg_bg, masks, depth. The above number reflects images which have been merged to common folder which is currently in progress but generation is completed.

##### fg_bg : 160x160x3
##### fg :  80 max size and min size is resized based on aspect ratio
##### mask: 160x160
##### depth images: 160x160


### The total size of the dataset :-
9182546 Output/ 6446124 Output/OverlayedImages/ 1119541 Output/OverlayedMasks/ 1616796 Output/DepthImage/

### Mean/STD values for your fg_bg, masks and depth images
#### fg_bg :- (BGR format) Mean: - [0.3234962448835791, 0.3776562499540454, 0.4548452917585805] stdDev: - [0.22465676724491895, 0.2299902629415973, 0.23860387182601098]

#### masks :- (BGR format) Mean: - [0.07863663756127236, 0.07863663756127236, 0.07863663756127236] stdDev: - [0.2541994994472449, 0.2541994994472449, 0.2541994994472449]

#### depth :- (BGR format) Mean: - [0.2943823440611593, 0.2943823440611593, 0.2943823440611593] stdDev: - [0.15619204938398595, 0.15619204938398595, 0.15619204938398595]

### Explain how you created your dataset

#### how were fg created with transparency 
We mainly downloaded images from internet without background, for some images we extracted foreground by using background removal technique in PowerPoint as shown in lecture.

#### how were masks created for fgs 
We figure out that mask images are nothing but alpha channels of images. So we extracted masks using following code image = cv2.imread("Foregroundimg.png", cv2.IMREAD_UNCHANGED) imagealpha = image[:,:,3] cv2.imwrite("ForegroundMask.jpg", imagealpha)

#### how did you overlay the fg over bg and created 20 variants
first all background images were resized to 224*224
all foreground images were resized to 80(max side) and other side was reshaped as per aspect ratio
images were randomly placed by choosing starting x,y randomly on background, but also making sure that foreground image doesnot go out of background image.
Code for generation of data is mentioned in DataGeneration.py

#### how did you create your depth images?
Although we used mostly same code as given in assignment for depthimage, we need to modify code to save images.
We have modified code in utils.py and test.py to save images directly to drive
