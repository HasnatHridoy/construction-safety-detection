![V-Safe: Vison based construction safety detection](readme_glossaries/banner.png)


<div>

<a href="https://huggingface.co/hasnatz/v-safe-rf-detr"><img src="readme_glossaries/hug.png"></a> &nbsp;&nbsp;
<a href="https://universe.roboflow.com/vision-works/cosntruction-safety-3-baqry"><img src="readme_glossaries/robo.png"></a>

</div>

## Problem Statement

Safety is a crucial aspect of any construction site. To prevent unexpected incidents or even life-threatening accidents, ensuring safety in construction environments is essential. In this project, our goal is to find solutions for enhancing construction safety using computer vision.

We have scraped around 12,000 images from the internet to train multiple object detection models that can identify safety equipment on construction sites.

## The goals

For this project, our objectives are as follows:

- To detect six safety items — helmet, gloves, mask, safety glasses, safety boots, and vest — along with six other common construction-related objects, including trucks, excavators, cranes, ladders, and workers.
- To train and evaluate different types of object detection models in order to identify the most effective one for this task.

## Results

We trained three different object detection models and selected **RF-DETR Medium** as the most suitable for our detection task. The model achieved a **Mean Average Precision (mAP@50–90)** score of **0.72**. Below are some annotated results produced by our model.

> **Note:** For detailed performance metrics and annotated results of each model, please refer to the **"Reproduction of this Project"** section.

<div align='center'>

<img src="readme_glossaries/image_result_1.png" alt="Annotated result image 1" width="50%">

https://github.com/user-attachments/assets/076cdad5-4f0a-4ea5-ac20-ae64dc2c3819

<img width="65%" alt="Screenshot 2025-10-07 120118" src="https://github.com/user-attachments/assets/ed13074a-0d2d-4ad0-bfb8-663232323fe6" />


</div>

