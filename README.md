![V-Safe: Vison based construction safety detection](readme_glossaries/banner.png)


<div>

<a href="https://huggingface.co/hasnatz/v-safe-rf-detr"><img src="readme_glossaries/hug.png"></a> &nbsp;&nbsp;
<a href="https://universe.roboflow.com/vision-works/cosntruction-safety-3-baqry"><img src="readme_glossaries/robo.png"></a>

</div>

## Problem Statement

Safety is a crucial aspect of any construction site. To prevent unexpected incidents or even life-threatening accidents, ensuring safety in construction environments is essential. In this project, our goal is to find solutions for enhancing construction safety using computer vision.

We have scraped around 10,000 images from the internet and after cleaning we used 9600 images to train multiple object detection models that can identify safety equipment on construction sites.

## The goals

For this project, our objectives are as follows:

- To detect six safety items — helmet, gloves, mask, safety glasses, safety boots, and vest — along with six other common construction-related objects, including trucks, excavators, cranes, ladders, and workers.
- To train and evaluate different types of object detection models in order to identify the most effective one for this task.

## Results

We trained three different object detection models and selected **RF-DETR Medium** as the most suitable for our detection task. The model achieved a **Mean Average Precision (mAP@50–95)** score of **0.72**. Below are some annotated results produced by our model.

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

<a target="_blank" href="https://colab.research.google.com/github/HasnatHridoy/construction-safety-detection/blob/main/Notebooks/Data%20Collection/image_scraping_%26_cleaning.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

<br>

<br>

> Note: You need to set up the Kaggle API key. <br>
> - Go to your Kaggle Account Settings → API → Create New API Token. <br>
> - This will download a file named kaggle.json. <br>
> - Upload this file to your notebook. <br>
> - Further instructions are provided within the notebook. 

<br>

#### Image labeling

We used Grounded SAM 2 for the image auto labeling then we uploaded the image on the Roboflow.

<a target="_blank" href="https://colab.research.google.com/github/HasnatHridoy/construction-safety-detection/blob/main/Notebooks/Labeling/Image_labelling.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

> Note: You need Roboflow account to upload the image to Roboflow
> - create a Roboflow account
> - create a project in your workspace
> - click 'settings' -> 'API key' to get your API
> - open your project and check the url in the browser it should look like this https://app.roboflow.com/your_workspace_name/your_project_name/other
> - copy your workspace and project name
> - Further instructions are provided within the notebook.

<br>

#### Annotation review

After uploading the images you should check the images for any mislabeled image to remove them. So go to your project in Roboflow and review the images.

#### Dataset creation

- Go to your project on Roboflow.
- Select the 'Version' from the sidebar.
- Select "Create new version"
- Adjust your "Train/Test Split" (we have used 78% for train, 20% for validation & 2% for testing.)
- Adjust your 'Preprocessing' step. (For preprocessing we have used Auto-Orient and Resize (Fit within 640×640) to decrease training time and increase performance by applying these transformations to all images in the dataset.)
- Setup you data 'Augmentation' (For image augmentation we have used Flip (Horizontal, Vertical), 90 
∘
  Rotate (Clockwise, Counter-Clockwise, Upside Down), Crop (8% Min Zoom, 23% Max Zoom), Rotation (Between ±13 
∘
 ), Shear (±14 
∘
  Horizontal, ±4 
∘
  Vertical), Grayscale (Apply to 7% of images), Hue (Between ±19 
∘
 ), Saturation (Between ±30%), Brightness (Between ±19%), Exposure (Between ±15%), Blur (Up to 2.6px), and Noise (Up to 1.88% of pixels).)
- Click the 'Create' and select 'Maximum Version Size' (we use 2x).

#### Dataset uses 

Before using the dataset you must use the proper format of the dataset. We have used Yolov11, Yolov12 and RFDETR models for our training so we used the 'yolo11 format' for Yolo11 model, 'yolo12 format' for the Yolo12 model and "coco" format for the RFDETR model. To select your proper format of the dataset:
- After creating the dataset click 'Download dataset' and select proper format.
- Select 'Show download code' then select "continue"

> Note: You don't need to create whole dataset for different format. Just download the format you need.

<br>

### Model Training

For this project we have used three model two from Yolo family (YOLOv11n & YOlOv12s) and one from the RF-DETR family (RF-DETR medium). Bellow are the training notebook of the models. All models are trained on T4 GPU on Kaggle.

> Note:
> - Don't Forget to turn on the acclerator from the sidebar.
> - Use the 'save version' -> 'save & run all (comit)' to run as background instance

<br>

YOLOv11n: We have trained this model for 80 epoch. After training we have achived mAP-50:95 (Mean Average Precision averaged over multiple Intersection over Union (IoU) thresholds.)
0.613

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/HasnatHridoy/construction-safety-detection/blob/main/Notebooks/Model%20training/yolov11n-training.ipynb)

<div align='center'> 

<img width="75%" alt="yolov11n_training_result" src="https://github.com/user-attachments/assets/1730343c-9414-4eeb-8cc2-efbfb1b4b6fa" />

</div>

<br>

<br>

YOLOv12s: We have trained this model for 65 epoch. After training we have achived mAP-50:95 value of 0.625

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/HasnatHridoy/construction-safety-detection/blob/main/Notebooks/Model%20training/yolov12s-training.ipynb)

<div align='center'>

<img width="75%" alt="yolov12s_training_result" src="https://github.com/user-attachments/assets/3a21bb4e-b608-440d-92ea-37f28ac8fa1b" />

</div>

<br>

<br>

RF-DETR Medium: We have trained this model for 10 epoch. After training we have achived mAP-50:95 of 0.72

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/HasnatHridoy/construction-safety-detection/blob/main/Notebooks/Model%20training/rf-detr-medium-training.ipynb)

<div align='center'>
  
<img width="75%" alt="image" src="https://github.com/user-attachments/assets/c540f293-44be-40e1-8f0a-f0e00a2843f7" />

</div>

<br>

### Model testing & compairson

The comparison of all models, based on performance on our testing dataset, is provided below

<div align='center'>
  
<img width="75%" alt="image" src="https://github.com/user-attachments/assets/39c1a979-fcae-4df6-92ef-da6c3b9d2806" />

*fig: Comparison of three models in different metrics*

</div>

<div align='center'>

<img width="75%" alt="Screenshot 2025-10-09 110707" src="https://github.com/user-attachments/assets/a08acfa2-989d-4ce6-812d-c2f12a6dac62" />

*fig: Comparison Table for Three Models Across Different Metrics.*

</div>

<br>

<br>

<div align='center'>

<img width="50%" alt="download (7)" src="https://github.com/user-attachments/assets/ecd22ef0-69a8-41da-8745-2011732b5b83" />

*fig: Confusion matrix for RF-DETR Medium*

<br>

<img width="50%" alt="confusion_matrix_normalized for yolov12s" src="https://github.com/user-attachments/assets/fcb6a143-6202-4e84-9c05-97f7e29983e9" />

*fig: Confusion matrix for YOLOv12s*

<br>

<br>

<img width="50%" alt="confusion_matrix_normalized (3)" src="https://github.com/user-attachments/assets/04557c33-7ae4-4f20-8096-7bb94e644bf3" />

*fig: Confusion matrix for YOLOv11n*

<img width="75%" alt="Group 38" src="https://github.com/user-attachments/assets/8c63be2a-484d-4418-83d3-13f2a4c8c9be" />

*fig: Inference compairson for three model*


</div>


#### Model selection for deployment

After testing, we decided to select **RF-DETR (Medium)** for deployment. As shown in the testing section, this model outperformed both **YOLOv11n** and **YOLOv12s.**

Interestingly, we observed that *all models were trained on single-class object instances but evaluated on multi-class object predictions. The YOLO models performed poorly when detecting multiple objects from a single image, even with very low confidence thresholds. Although we included a few multi-class examples during YOLOv12s training, it was not sufficient to improve its performance.*

*In contrast, the RF-DETR model—despite being trained on fewer images and without any image augmentation (due to limited VRAM)—showed a significant improvement in detecting multiple classes within a single image. We suspect that the model’s architecture played a crucial role in this outcome.*

While the inference speed of YOLO models was notably faster, we believe that RF-DETR’s inference performance can be improved by using a larger batch size during video-based inference. 

#### Model conversion

The model’s PyTorch checkpoint was converted to ONNX format using FP32 precision and a static input shape. 
Both onnx and checkpoint has uploaded to the <a href="https://huggingface.co/hasnatz/v-safe-rf-detr">Huggingface model hub</a>

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/HasnatHridoy/construction-safety-detection/blob/main/Notebooks/Model%20export/conversion-to-onnx-and-export-to-huggingface.ipynb)

## Model Deployment

We have deployed the model on <a href='https://huggingface.co/spaces/hasnatz/v-safe-demo'> Huggingface Space </a>. The implementation can be found in Huggingface Space and also in the `Deployment` folder.

<br>

<div align='center'>

  <img width="75%" alt="image" src="https://github.com/user-attachments/assets/c6fc50c4-d036-4d7b-82f0-a31fc529face" />

</div>

## API integration with GitHub Pages

The deployed model API has been integrated with our GitHub Pages website. Implementation details related to the site are available in the `github_page_asset` folder.
