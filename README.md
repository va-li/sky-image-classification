# Applied Deep Learning 2024WS Project Proposal

Topic: *Image Classification*

Project Type: *Bring your own data*

Table of Contents:

- [Assignment 1 - Proposal](#assignment-1---proposal)
  - [Project idea and approach](#project-idea-and-approach)
  - [Dataset](#dataset)
  - [Work-breakdown Structure (85-95 hours)](#work-breakdown-structure-85-95-hours)
  - [References](#references)
- [Assignment 2 - Hacking](#assignment-2---hacking)
- [Setup and Installation](#setup-and-installation)
  - [Training](#training)
  - [Webapp](#webapp)
- [Intermediate Updates](#intermediate-updates)
  - [Update 2024-12-17](#update-2024-12-17)
  - [Update 2024-11-21](#update-2024-11-21)
  - [Update 2024-11-20](#update-2024-11-20)
  - [Update 2024-10-29](#update-2024-10-29)

---

# Assignment 1 - Proposal

## Project idea and approach

**Multi-label classification of image artifacts in hemispherical sky images.** The images come from a dataset of approximately 10,000 images collected from my all-sky camera prototype during my master's thesis. I am at the end of this thesis, which aims to provide the most accurate possible estimation of diffuse sky radiation per pixel from hemispherical sky images for solar energy applications. Unfortunately, image artifacts such as dirt on the optics, dew, and similar issues cause incorrect radiation estimates, but many of them can be corrected by cleaning the optics. Further, I need to distinguish clear-sky from clouded-sky images, as the clear-sky ones will be used for radiometric calibration of the camera. Because the camera is built from a Raspberry Pi 4, it is possible to use a neural network directly on the camera hardware. To allow on-demand maintenance of the camera capturing these images, I would like to work on classifying these artifacts directly on the device.

My goal for this project is to label at least half of the dataset (~5.000 images) and train an image classifier that runs on the Raspberry Pi 4 and can classify a sky image in no more than a few seconds. Up to four labels should be possible simultaneously for a sky image: clouds, soiling, water droplets inside (from dew), and water droplets outside (from rain). This could enable me to make a shift from regular to on-demand maintenance of the camera and provides me with the opportunity to get more hands-on experience with deep learning.

Looking into the literature, I found two recent papers that will be relevant for my project. They address the challenges of deploying deep learning models on resource-limited devices like the Raspberry Pi, which is the core of the camera prototype. The first paper presents TripleNet, a lightweight convolutional neural network designed specifically for efficient image classification on the Raspberry Pi, showing the potential for milli-second inference time and below 100 MB memory usage (1). The second paper focuses on the real-time classification of sky conditions using deep learning and edge computing, aligning directly with my project's aim of automating the detection of artifacts in hemispherical sky images. It demonstrates the feasibility of automating sky image analysis on low-cost hardware (2).

## Dataset

The dataset consists of 9755 daytime hemispherical sky images collected from May to October 2024 with two all-sky camera prototypes in Vienna, Austrian and Sophia Antipolis, France. An example of such an image, without artifacts, taken on 23.08.2024 at 11:30 in Sophia Antipolis can be seen below.

<img src="dissemination/images/2024-08-23T09-30-00-006516+00-00.png" width="400" alt="Hemispherical Sky Image">

Each image is fused from the original raw capture of five increasing exposures into one high dynamic range image (HDR) to avoid overexposed pixels from the bright sunlight and capture details in clouds at the same time. The images from one camera have a resolution of 884x850 pixels and 892x864 pixels from the other camera. Below are examples for each of the five classes that we want to detect: **clear sky**, **clouds**, **soiling**, **water droplets inside** (from dew), and **water droplets outside** (from rain).

| Clear Sky                                                               | Clouds                                                                    | Soiling                                                                                                                    | Water droplets inside                                                                         | Water droplets outside                                                               |
|-------------------------------------------------------------------------|---------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| ![Clear Sky](dissemination/images/2024-08-23T09-30-00-006516+00-00.png) | ![Clouds](dissemination/images/2024-08-24T10-00-00-006417+00-00.png)      | ![Soiling](dissemination/images/2024-07-21T14-30-00-006321+00-00_annotated.png)                                            | ![Water droplets inside](dissemination/images/2024-09-17T12-45-00-006205+00-00_annotated.png) | ![Water droplets outside](dissemination/images/2024-09-04T12-45-00-006141+00-00.png) |
| Clear sky. Mutually exclusive with *Clouds* class.                      | Clouds are visible in the sky. Mutually exclusive with *Clear Sky* class. | Dirt on the dome protecting the lens is visible, mostly by reflection around the sun, but also in other areas of the dome. | Water droplets from dew inside of the dome protecting the lens are visible.                   | Water droplets outside from rain on the dome protecting the lens are visible.        |

Looking at the third and fourth images we can see that the classes can overlap (clouds can be present with rain or dew). This is why we need a multi-label classification approach.

## Work-breakdown Structure (85-95 hours)

### Dataset Collection (25-35 hours)

- Collecting and preprocessing the images, including merging exposures into HDR images and rescaling to common resolution: **5 hours**
- Manual labeling of the dataset (at least ~5,000 images) using CVAT (3) into four classes: **20-30 hours**

### Designing, Building and Training the Network (25 hours)

- Adapt TripleNet [1] for multi-label classification (currently set up fro single label classification on CIFAR10): **10 hours**
- Train the model and experiment with different hyperparameters (not including actual training time): **15 hours**

### Building the Application (15 hours)

- Test the model on the Raspberry Pi 4 for performance (accuracy and speed): **5 hours**
- Develop a basic web interface for uploading and classifying images on the Raspberry Pi 4: **10 hours**

### Writing the Final Report and Preparing the Presentation (20 hours)

- Creating the report and with proposal, methodology, results, and insights: **10 hours**
- Preparing the presentation, recording, and uploading it: **10 hours**

## References

(1) Efficient convolutional neural networks on Raspberry Pi for image classification ([DOI: 10.1007/s11554-023-01271-1](https://doi.org/10.1007/s11554-023-01271-1))

(2) Real-Time Automatic Cloud Detection Using a Low-Cost Sky Camera ([DOI 10.3390/rs12091382](https://doi.org/10.3390/rs12091382))

(3) CVAT: Computer Vision Annotation Tool ([CVAT on GitHub](https://github.com/cvat-ai/cvat))

---

# Assignment 2 - Hacking

Error metric: [**Jaccard Score** or **Intersection over Union (IoU)**](https://en.wikipedia.org/wiki/Jaccard_index)

It is a measure of the similarity between two sets and defined as the size of the intersection divided by the size of the union of the sets. A high Jaccard Score means the model only predicts the correct labels and no false positives.

In the context of my multi-label classification problem, I calculate the Jaccard Score for each sample and then average over all samples. The rational behind choosing the Jaccard Score is that it allows to measure how well the model predicts multiple labels for a single image, and it is comparable for a different number of labels per image.

|                         | Mean Jaccard Score |
|-------------------------|-------------------:|
| Target                  |               0.75 |
| **Achieved** (test set) |           **0.85** |

Despite trying data augmentation and higher resolution images than what MobileNetV3 supports by default, the best performing model on the validation set was the original MobileNetV3 model trained on 224x224 images without data augmentation. 

Further, I also looked at the macro precision and recall, which are the average of the precision and recall for each class, to see if the model is "fair" in predicting all classes.

|                         | Macro Precision | Macro Recall |
|-------------------------|----------------:|-------------:|
| **Achieved** (test set) |        **0.89** |     **0.57** |

And in a bit more detail for each class in test set:

| Class     | Precision (one-vs-rest) | Recall (one-vs-rest) | **# labels** |
|-----------|------------------------:|---------------------:|-------------:|
| clouds    |                   0.927 |                0.998 |          470 |
| rain      |                   0.875 |                0.867 |          113 |
| dew       |                   0.992 |                0.646 |          192 |
| clear sky |                   0.929 |                0.260 |           50 |
| soiling   |                   0.093 |                0.714 |           54 |


Time spent on each task:

- Dataset Collection:
  - Collecting and preprocessing the images: **3 hours**
  - Manual labeling of the dataset: **18 hours**
- Designing, Building and Training the Network:
    - Setting up the training pipeline: **2 hours**
    - Adapting MobileNetV3 for multi-label classification: **1 hour**
    - Training the model and experimenting with hyperparameters and debugging: **47 hours**
- Building the Application:
    - Testing the model on the Raspberry Pi 4: **4 hours**
    - Developing a basic web interface for uploading and classifying images: **14 hours**

**Total: 89 hours**

---

# Setup and Installation

## Training

Required software:

- Python 3.11
- Python 3.11 venv
- NVIDIA CUDA Toolkit
- Python packages: numpy, pandas, scikit-learn, scikit-image, tqdm, matplotlib, tabulate, torch, torchvision, torchaudio

Either install them manually or run [install_train_dependencies.sh](./install_train_dependencies.sh) to install the system requirements, create a virtual environment in the project directory, and install the Python packages all at once.

There is a python requirement file in the repository, but despite being created after the installation of all dependencies with the install script, it does not work for me to install the python dependencies with `pip install -r requirements_train.txt`. I think this is because the right PyTorch version is not available on PyPi, but only on via the PyTorch index.

The dataset is not included in the repository, but can be provided on request. The dataset should be placed in [data/](./data/).

To train the model, go into [src/](./src/) and run [train_model.py](./src/train_model.py).

The training script will save all training data and the best model in a folder under `data/training_runs/` with a timestamp as the folder name. The training script will also save the model with the best jaccard score and the training history as a CSV file and plot in the training run folder.

To test a model separately from the training loop on the test set, go into [src/](./src/) agian run [test_model.py](./src/test_model.py). The script will load the best model from the training run folder and test it on the test set. The results will saved in the training run folder.

## Webapp

Required software:

- Python 3.11
- Python 3.11 venv

Go into [src/webapp/](./src/webapp/) and run [install_webapp_dependencies.sh](./src/webapp/install_webapp_dependencies.sh) to install the system requirements, create a virtual environment in the project directory, and install the Python packages all at once.

Then pick the best model from one of the training runs (`best_model.pth`) and place it in [src/webapp/classifier](./src/webapp/classifier/).

To start the webapp, run [run.sh](./src/webapp/run_webapp.sh). The webapp will be available at [http://localhost:8000](http://localhost:8000).

![Webapp Screenshot 1](dissemination/images/webapp_screenshot_2024-11-21_1.png)

---

# Intermediate Updates

## Update 2024-12-17

### Dataset

- 5201 (53%) images are labeled, 4554 (47%) images are left to label, 79 (1%) were excluded (9780 images in total).
- 1910 (37%) images are labeled with one label, 2493 (48%) with two labels, 785 (15%) with three labels and  13 (<1%) with four labels.
- Previously, the model was trained by shuffling the data first and then splitting it into training, validation and test sets. This was a bad choice, because images from the same day were in both the training, validation and test sets. This caused data leakage and the model to perform very well on the test set (see high metrics in the previous updates). To fix this, I now split the data without shuffling. Since they are ordered by capture timestamp, no data leakage should occur and the metrics should be more realistic.

Number of labels assigned to images per class:

|     Class     | # labels |
|:-------------:|---------:|
|  **clouds**   |     4172 |
|   **rain**    |      537 |
|    **dew**    |     1437 |
| **clear sky** |     1029 |
|  **soiling**  |     2128 |
|   **Total**   | **9303** |

Number of images per label combination (only existing label combinations shown):

|  clouds   | rain  |  dew  | clear sky | soiling | # images |
|:---------:|:-----:|:-----:|:---------:|:-------:|---------:|
|           |       |       |   **X**   |         |      487 |
|           |       |       |   **X**   |  **X**  |      328 |
|           |       | **X** |   **X**   |         |       79 |
|           |       | **X** |   **X**   |  **X**  |      118 |
|           | **X** |       |   **X**   |         |        4 |
|           | **X** |       |   **X**   |  **X**  |        2 |
|           | **X** | **X** |   **X**   |         |       11 |
|   **X**   |       |       |           |         |     1423 |
|   **X**   |       |       |           |  **X**  |     1213 |
|   **X**   |       | **X** |           |         |      616 |
|   **X**   |       | **X** |           |  **X**  |      400 |
|   **X**   | **X** |       |           |         |      253 |
|   **X**   | **X** |       |           |  **X**  |       54 |
|   **X**   | **X** | **X** |           |         |      200 |
|   **X**   | **X** | **X** |           |  **X**  |       13 |
| **Total** |       |       |           |         | **5201** |


### Models

- The original MobileNetV3 model takes images of size 224x224 as input. I wanted to try if the model performs better with higher resolution images. Therefore, I adapted the model with an additional convolutional layer in the beginning to handle 448x448 images. The model was trained on the same data as before, but with the higher resolution images. Results are shown below.
- Also, I tried augmenting the data with random rotations, flips, and color jittering. Results are shown below.
- Only very late in the project I realized that caching the entire dataset (2.6GB on disk) in memory is not only possible, but reduces my training time massively. Without caching a single epoch took around 7 minutes. After caching, every epoch after the first took only a few seconds.

Results of the different experiments (more hyperparameters were tested, but only the best performing models are shown):

| Rank | Data Augmentation | Image Size | Mean Jaccard (val) | Macro Precision (val) | Macro Recall (val) | Train time (minutes) |
|-----:|:------------------|-----------:|-------------------:|----------------------:|-------------------:|---------------------:|
|    1 | False             |        224 |           **0.82** |              **0.88** |           **0.70** |                 5.28 |
|    3 | True              |        224 |               0.82 |                   0.8 |               0.71 |                 9.49 |
|    5 | False             |        448 |               0.75 |                  0.77 |               0.61 |                 4.33 |
|    6 | True              |        448 |               0.72 |                  0.78 |               0.61 |                 9.04 |

## Update 2024-11-21

- First useful version of the model is trained.
- Training was done with fine-tuning the MobileNetV3Large model and using data augmentation.
- Training was done on 3222 images, validated on 402 images. Test set not yet used.
- Training time was around 2h40m on my laptop's NVIDIA GeForce MX150.
- The model achieves on the validation set a subset accuracy of 89% and a mean Jaccard score of 94%, with a macro averaged precision of 95% and recall of 93% over all five classes.
- Added basic webapp to classify images using the trained model. The webapp is available in [./src/webapp/](./src/webapp/).

**Example of model output on unseen images**:

| Image                                                                   | Prediciton                                                        |
|-------------------------------------------------------------------------|-------------------------------------------------------------------|
| ![](dissemination/images/daedalus_2024-06-01T09-30-00-006100+00-00.jpg) | `clouds: 1.00 rain: 0.93 soiling: 0.36 dew: 0.32 clear sky: 0.00` |
| ![](dissemination/images/daedalus_2024-06-26T07-00-00-006205+00-00.jpg) | `clouds: 1.00 dew: 0.98 soiling: 0.70 rain: 0.00 clear sky: 0.00` |
| ![](dissemination/images/ikarus_2024-05-22T15-57-58-017680+00-00.jpg)   | `clear sky: 0.99 clouds: 0.01 soiling: 0.00 dew: 0.00 rain: 0.00` |

**Screenshots of the webapp**:

![Webapp Screenshot 1](dissemination/images/webapp_screenshot_2024-11-21_1.png)

## Update 2024-11-20

- 4026 (41%) images are labeled, 5700 (58%) images are left to label.
- 54 (1%) images have been excluded from the dataset, because they include people or other objects (spiders, birds, etc.) or they are of poor quality due to high noise.
- A first version of a multi-label classifier based on MobileNetV3 has been fine-tuned on the labeled images. The results are promising, but the model is not yet generalizing well to unseen images.

**Random sample of labeled images**:

![Random Sample of Labeled Images](dissemination/images/labeled_data_sample_2024-11-20.png)

**Random sample of excluded images**:

![Random Sample of Excluded Images](dissemination/images/excluded_data_sample_2024-11-20.png)

## Update 2024-10-29

- Created a dataset by merging raw exposures into HDR images. They amount to 9780 images.
- Created a custom PyTorch dataset class [SkyImageMultiLabelDataset](./src/dataset.py).
- Successfully followed pytorch guide on [Real Time Inference on Raspberry Pi 4](https://pytorch.org/tutorials/intermediate/realtime_rpi.html) to test running a MobileNetV3 model on the Raspberry Pi 4.
- Decided to use the MobileNetV3 model as a starting point for the project, instead of TripleNet. Reasons: weights for MobileNetV3 are available in PyTorch, but not for TripleNet; MobileNetV3 expects an input size of 224x224, which is closer to the resolution of the images in the dataset, while TripleNet expects 32x32 images.
