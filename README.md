# Image2Depth

## Objective
Depth prediction from monocular images for Autonomous Driving.

## Solution Approaches


## Dataset
We have used [Apollo Scenes](http://data.apollo.auto/?locale=en-us&lang=en) dataset and a custom dataset created by us using [AirSim Engine by Unity and Microsoft] in this project.
Apollo dataset consists of ________ images. 
Custom dataset created by us consists of ________ images.

## Pre-processing
We clipped the depth image at 80 instead of 165. This is done to focus more on the depth upto 80m.
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

<p align='center'>
  <img src='./outputs/readme_out/input_1.png' style="width: 300px;" />
  <img src='./outputs/readme_out/gt_1.png' style="width: 300px;" />
  <img src='./outputs/readme_out/output_1.png' style="width: 300px;" />
</p>

<br>
	<br>
<b> Here is the plot of the losses:</b>
<br>
<br>

<p align='float'>
  <img src='./outputs/readme_out/loss.png' style="width: 300px;" />
</p>
L1 is 
L2 is
L3 is 
L4 is


### Some observations



## Requirements 


## Instructions
 

## References



