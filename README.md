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

## Reproduction of this project

In this section we will provide the step by step guide for reproducing this project

### Workflow

<div align='center'>

<img width="65%" alt="Wrokflow simple" src="https://github.com/user-attachments/assets/07c32f4b-f660-4d79-bea3-7ad005bef807" />

*fig.: High level overview of the workflow*

<br>

<img width="1533" height="1421" alt="workflow_detailed" src="https://github.com/user-attachments/assets/9d836ab6-9258-46fd-b0cb-10657e1bbfbd" />

*fig.: Detailed workflow map*

</div>

<br>

<br>

#### Image scraping & cleaning

The images were scraped from Google, Bing, and DuckDuckGo using iCrawler and ddgs. We then combined the collected images, removed duplicates, and kept only those in .jpg format.
And upload the unannotated dataset to the Kaggle.

> Note: You need to set up the Kaggle API key. <br>
> Go to your Kaggle Account Settings → API → Create New API Token. <br>
> This will download a file named kaggle.json. <br>
> Upload this file to your notebook. <br>
> Further instructions are provided within the notebook. 
