# Applied Deep Learning 2024WS Project Proposal

Topic: *Image Classification*

Project Type: *Bring your own data*

## Project idea and approach

**Multi-label classification of image artifacts in hemispherical sky images.** The images come from a dataset of approximately 10,000 images collected from my all-sky camera prototype during my master's thesis. I am at the end of this thesis, which aims to provide the most accurate possible estimation of diffuse sky radiation per pixel from hemispherical sky images for solar energy applications. Unfortunately, image artifacts such as dirt on the optics, dew, and similar issues cause incorrect radiation estimates, but many of them can be corrected by cleaning the optics. Further, I need to distinguish clear-sky from clouded-sky images, as the clear-sky ones will be used for radiometric calibration of the camera. Because the camera is built from a Raspberry Pi 4, it is possible to use a neural network directly on the camera hardware. To allow on-demand maintenance of the camera capturing these images, I would like to work on classifying these artifacts directly on the device.

My goal for this project is to label at least half of the dataset (~5.000 images) and train an image classifier that runs on the Raspberry Pi 4 and can classify a sky image in no more than a few seconds. Up to four labels should be possible simultaneously for a sky image: clouds, soiling, water droplets inside (from dew), and water droplets outside (from rain). This could enable me to make a shift from regular to on-demand maintenance of the camera and provides me with the opportunity to get more hands-on experience with deep learning.

Looking into the literature, I found two recent papers that will be relevant for my project. They address the challenges of deploying deep learning models on resource-limited devices like the Raspberry Pi, which is the core of the camera prototype. The first paper presents TripleNet, a lightweight convolutional neural network designed specifically for efficient image classification on the Raspberry Pi, showing the potential for milli-second inference time and below 100 MB memory usage (1). The second paper focuses on the real-time classification of sky conditions using deep learning and edge computing, aligning directly with my project's aim of automating the detection of artifacts in hemispherical sky images. It demonstrates the feasibility of automating sky image analysis on low-cost hardware (2).

## Dataset

The dataset consists of 9755 daytime hemispherical sky images collected from May to October 2024 with two all-sky camera prototypes in Vienna, Austrian and Sophia Antipolis, France. An example of such an image, without artifacts, taken on 23.08.2024 at 11:30 in Sophia Antipolis can be seen below.

<img src="dissemination/images/2024-08-23T09-30-00-006516+00-00.png" width="400" alt="Hemispherical Sky Image">

Each image is fused from the original raw capture of five increasing exposures into one high dynamic range image (HDR) to avoid overexposed pixels from the bright sunlight and capture details in clouds at the same time. The images from one camera have a resolution of 884x850 pixels and 892x864 pixels from the other camera. Below are examples for each of the five classes that we want to detect: **clear sky**, **clouds**, **soiling**, **water droplets inside** (from dew), and **water droplets outside** (from rain).

| Clear Sky | Clouds | Soiling | Water droplets inside | Water droplets outside |
|-----------|--------|---------|-----------------------|------------------------|
| ![Clear Sky](dissemination/images/2024-08-23T09-30-00-006516+00-00.png)| ![Clouds](dissemination/images/2024-08-24T10-00-00-006417+00-00.png) | ![Soiling](dissemination/images/2024-07-21T14-30-00-006321+00-00_annotated.png) | ![Water droplets inside](dissemination/images/2024-09-17T12-45-00-006205+00-00_annotated.png) | ![Water droplets outside](dissemination/images/2024-09-04T12-45-00-006141+00-00.png) |
| Clear sky. Mutually exclusive with *Clouds* class. | Clouds are visible in the sky. Mutually exclusive with *Clear Sky* class. | Dirt on the dome protecting the lens is visible, mostly by reflection around the sun, but also in other areas of the dome. | Water droplets from dew inside of the dome protecting the lens are visible. | Water droplets outside from rain on the dome protecting the lens are visible. |

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
