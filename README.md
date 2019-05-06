# Image2Depth

## Objective
Depth prediction from monocular images for Autonomous Driving.

## Solution Approach


## Dataset
We have used [Apollo Scenes](http://data.apollo.auto/?locale=en-us&lang=en) dataset and a custom dataset created by us using <b> AirSim Engine by Unity and Microsoft</b> in this project.
<br>
Apollo dataset consists of 23k images.
<br>
Custom dataset created by us consists of 99k images.

## Pre-processing
We clipped the depth image at 80 instead of 165. This is done to focus more on the depth upto 80m as in autonomous driving vision upto 80m is good enough for desicion making.
<br>
We also removed the upper half of the both rgb and depth image to remove sky from the scene.

## Model Specifications

Model architecture used by us is:-
<p align='center'>
  <img src='./outputs/readme_out/model.png' alt='model'/>
</p>

Here, DCNN block mentioned is:-
<p align='center'>
  <img src='./outputs/readme_out/DCNN.png' alt='dcnn'/>
</p>


## Some Results
			Input				Ground Truth				Output
<p align='center'>
  <img src='./outputs/readme_out/input_1.png' style="width: 300px;" />
  <img src='./outputs/readme_out/gt_1.png' style="width: 300px;" />
  <img src='./outputs/readme_out/output_1.png' style="width: 300px;" />
</p>
<p align='center'>
  <img src='./outputs/readme_out/input_2.png' style="width: 300px;" />
  <img src='./outputs/readme_out/gt_2.png' style="width: 300px;" />
  <img src='./outputs/readme_out/output_2.png' style="width: 300px;" />
</p>
<p align='center'>
  <img src='./outputs/readme_out/input_3.png' style="width: 300px;" />
  <img src='./outputs/readme_out/gt_3.png' style="width: 300px;" />
  <img src='./outputs/readme_out/output_3.png' style="width: 300px;" />
</p>
<br>
	<br>
<b> Here is the plot of the losses:</b>
<br>
<br>

<p align='float'>
  <img src='./outputs/readme_out/loss.png' style="width: 300px;" />
</p>
<br>
L1 is absolute mean difference at 1/64th of original resolution
<br>
L2 is absolute mean difference at 1/16th of original resolution
<br>
L3 is absolute mean difference at 1/4th of original resolution
<br>
L4 is  absolute mean difference at original resolution

<br>
	<br>
<b> Here is the mean and standard deviation plot:</b>
<br>
<br>
<p align='float'>
  <img src='./outputs/readme_out/mean_plot.png' style="width: 300px;" />
</p>

### Some observations


### Postprocessing results
We applied gamma correction on the predicted depth images:

<br>
				Output Image					Output image with gamma correction
<p align='center'>
  <img src='./outputs/readme_out/original_o.png' style="width: 300px;" />
  <img src='./outputs/readme_out/post_o.png' style="width: 300px;" />
</p>

<b> Pseudo color result images </b> <br>
				Ground Truth					Original output
<p align='center'>
  <img src='./outputs/readme_out/gt_color.png' style="width: 300px;" />
  <img src='./outputs/readme_out/output_color.png' style="width: 300px;" />
</p>
				Output with gamma correction
<p>
  <img src='./outputs/readme_out/gamma_color.png' style="width: 300px;" />
</p>

## Requirements
The entire codebase is written in compatibility with tensorflow 2.0. for installing the other
python libraries please use the requiremets.txt file.
```
  $pip install -r requirements.txt
```
This text file contains all the requirements except [cv2 install guide](https://www.pyimagesearch.com/2018/05/28/ubuntu-18-04-how-to-install-opencv/).

To be able to train and validate your model, create 2 .json files of the names train.json and val.json ad they should be of the
following format:
```json
  {
    "<full path and name of input image>":"<full path and name of the corresponding ground truth image>"
  }
```



## Instructions
First and foremost please download the model checkpoint from [here]() and save the uncompressed files inside ./tmp

There are 2 things we can do here:
1 Train from scratch:

```
    $ python execute.py -m train
```
2 Test on images in the ./test folder
pleaser make sure that the image that you have provided is in 4:3 in W:H ratio. The given image will be first resized to (800,600) first and the input to the network is the IMAGE[H-288:H,W] and the result is the depth estimate stitched parallel to the given input.  
```
    $ python execute.py -m test
```


## References
[Densely connected convolutional network.](https://arxiv.org/abs/1608.06993)

[Appolo Dataset](http://apollo.auto/opendata.html)

[Microsoft AirSim for autonomous vehicle environments](https://github.com/Microsoft/AirSim)
