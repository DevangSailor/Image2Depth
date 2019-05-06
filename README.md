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
L1 is
<br> 
L2 is
<br>
L3 is
<br> 
L4 is

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
			Output Image				Output image with gamma correction
<p align='center'>
  <img src='./outputs/readme_out/original_o.png' style="width: 300px;" />
  <img src='./outputs/readme_out/post_o.png' style="width: 300px;" />
</p>

<b> Pseudo color result images </b>
			Ground Truth			Original output			Output with gamma correction
<p align='center'>
  <img src='./outputs/readme_out/gt_color.png' style="width: 300px;" />
  <img src='./outputs/readme_out/output_color.png' style="width: 300px;" />
  <img src='./outputs/readme_out/gamma_color.png' style="width: 300px;" />
</p>

## Requirements 


## Instructions
 

## References



