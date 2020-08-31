<img src="assets/gsoc.png" width="556px" height="112px"/>
    
## Google Summer of Code 2020: **Final Work Product**
### **Organisation**: TensorFlow
### **Mentor**
- Jaeyoun Kim ([@jaeyounkim](https://github.com/jaeyounkim))  
    
## **Aim**
State-of-the-art Deep Learning models for the TensorFlow Model Garden implemented from the most recent research papers and trained on the datasets they are benchmarked on. My aim is to add some of the most exciting models to the Model Garden to help make model prototyping much faster and promote research using TF 2.x. The project proposes models such as FineGan and DETR along with other exciting theoretical models. The TF Model Garden makes building new models with state-of-the-art performance much easier especially with the new additions to TF 2.x that have made every part of the model training pipeline much more efficient. The models I have proposed to add will be trained till they converge to the benchmarks in the paper and will have detailed documentation and lucid Colab demos to make them extensible and easy to use.  
# 
## **Implementation**
|Paper|Conference|Benchmark|
|---|---|---|
|[FineGAN](https://arxiv.org/abs/1811.11155)|CVPR ‘19 |[SoTA](https://github.com/kkanshul/finegan/) for CUB 128x128, CUB, Stanford Cars [Image Generation]|
|[DETR](https://arxiv.org/abs/2005.12872v3)|NeurIPS ‘20 |[SoTA](https://github.com/facebookresearch/detr/) for COCO Panoptic [Panoptic Segmentation]|
|[Expressivity of Batchnorm](https://arxiv.org/abs/2003.00152)|NeurIPS '20 |[Official Implementation](https://github.com/facebookresearch/open_lth) for CIFAR10 and ImageNet|
|[MUNIT](https://arxiv.org/abs/1804.04732)|ECCV '18 |[SoTA](https://github.com/nvlabs/MUNIT) for Cityscapes, Edge-to-Handbags [Image Translation]|
#
## **Progress**
|Evaluation|Task|Link|Status|Pull Request|
|---|---|---|---|---|
|E1| FineGAN Model |[Here](https://github.com/Vishal-V/tf-models/tree/master/finegan)| Complete |[ #8750](https://github.com/tensorflow/models/pull/8750)|
|E2| FineGAN Training Pipeline |[Here](https://github.com/Vishal-V/tf-models/tree/master/finegan)| Complete |[ #8750](https://github.com/tensorflow/models/pull/8750)|
|E3| FineGAN Benchmark |[Here](https://github.com/Vishal-V/tf-models/tree/master/finegan)| Partial |[ #8750](https://github.com/tensorflow/models/pull/8750)|
|E3| Expressivity of BN - Code|[Here](https://github.com/Vishal-V/tf-models/)|Complete| [ ]()|
|E3| Expressivity of BN - Notebook |[Here](https://github.com/Vishal-V/tf-models/)|Complete| [ ]()|
|E3| FineGAN - Notebook |[Here](https://github.com/Vishal-V/tf-models/tree/master/finegan)| Partial |[ #8750](https://github.com/tensorflow/models/pull/8750)|
|E3| Detection Transformer |[Here](https://github.com/Vishal-V/tf-models/)|WIP| [ ]()|
|E3| MUNIT |[Here](https://github.com/Vishal-V/tf-models/)|WIP| [ ]()|
#
#
## **Work Done**
### FineGAN
Implemented the 3-Stage FineGAN Architecture with a stage each for Background, Foreground Outline and Foreground Mask generation. This includes the PatchGAN Background scene detector to detect background only patches for the auxiliary discriminator in the Background stage. I also fixed several errors and added workarounds for assigning values to eager tensors and assigning zero weight to the tensors with cropped real bounding boxes. To improve the data loading performance, I added the `tf.data` pipelines training and evaluation paired images with the data augmentation and the modified masked bounding boxes. Model training provides results but is yet to be benchmarked [WIP].
  
- **Model**: https://github.com/Vishal-V/tf-models/tree/master/finegan  
- **Notebook**: [GitHub Link](https://github.com/Vishal-V/tf-models/blob/master/finegan/efficient%20trials.ipynb)
  
### Only batchNorm
  
Most scholars who complete the Stanford CS231N course attempt the final assignment to train a model on the TinyImageNet dataset without transfer learning. But, those scholars with resource constraints or only Google colab to fall back on find it difficult to train a decent model . This is a custom ResNet with 10x lesser parameters for image classification on the TinyImageNet dataset. The training strategy and data loading features are made efficient to enable training on Google colab. The model training uses `progressive resizing` of image sizes to enable the model to learn scale independent semantic features. The data is loaded using the `ImageDataGenerator` class.
- **Model**: https://github.com/Vishal-V/GSoC/tree/master/tiny_imagenet_custom_resnet/model
- **Instructions**: https://github.com/Vishal-V/GSoC/tree/master/tiny_imagenet_custom_resnet
- **Notebook**: [GitHub Link](https://github.com/Vishal-V/GSoC/blob/master/tiny_imagenet_custom_resnet/tiny_imagenet_custom_resnet.ipynb), [Colab Link](https://colab.research.google.com/drive/1SZLecFzKuU7TVCoCzTq285sEbjlT12A6)
  
### Detection Transformer (DETR)
Model created from the paper [[Arxiv Link](https://arxiv.org/pdf/1612.03242.pdf)] to train on the CUB 200 dataset for birds. There are 2 stages in the model. Stage 1 takes the latent space and a conditioning vector as input to generate 64x64 resolution images. The model uses pre-trained Char-RNN-CNN embeddings. Stage 2 generates 256x256 resolution images based off of the generated images from Stage 1 of the StackGAN.
- **Model**: https://github.com/Vishal-V/GSoC/tree/master/stack_gan
- **Instructions**: https://github.com/Vishal-V/GSoC/blob/master/stack_gan/README.md
  
### Age-cGAN
From the hype of FaceApp, the fully trained Age-cGAN model can be used to build your very own version. Although the model is currently training and the weights are yet to be uploaded, the script can be used to train an age conditional GAN on the Wiki cropped faces dataset. The training occurs in 3 phases: GAN training, latent vector approximation and latent vector optimization.
- **Model**: https://github.com/Vishal-V/GSoC/tree/master/face_app
- **Instructions**: https://github.com/Vishal-V/GSoC/tree/master/face_app/README.md
### Hyperparameter Tuning
This notebook was initially supposed to be a logo classifier akin to the Google I/O 19 demo, but the dataset was never made public. Hence, this is a Keras-Tuner notebook that shows 'hypertuning for humans' based on a comparison between a regular model and one marked for hyper-parameter tuning.
- **Notebook**: [GitHub Link](https://github.com/Vishal-V/GSoC/blob/master/keras_tuner/hyperparamter_search.ipynb), [Colab Link](https://colab.research.google.com/drive/15Mqrbtv5u39P9UOwIFjJWkpW6YsS7nA8)
### Mask R-CNN [WIP]
This model was supposed to be a migration of the most authoritative Mask R-CNN implementation available. Although the basic migartion was done, the very model uses a lot of graph functions that are still being fixed by me. I have translated a lot of the functions but hit quite a few roadblocks with NaNs. I plan on rebuilding the entire model from scratch being TF 2.0 native. This is the only task left to be completed.
- **Model**: https://github.com/Vishal-V/GSoC/tree/master/mask_rcnn
- **Migration Guide**: https://github.com/Vishal-V/GSoC/blob/master/mask_rcnn/README.md

## **Deliverables**
- [**Autoencoder**](https://github.com/Vishal-V/GSoC/tree/master/autoencoder/model) - [tensorflow/examples #68](https://github.com/tensorflow/examples/pull/68)
-  [**Autoencoder - Notebook**](https://github.com/Vishal-V/GSoC/tree/master/autoencoder/model) - [tensorflow/examples #68](https://github.com/tensorflow/examples/pull/68)
- [**StackGAN**](https://github.com/Vishal-V/GSoC/tree/master/stack_gan) -  [tensorflow/examples #77](https://github.com/tensorflow/examples/pull/77)
- [**Age-cGAN**](https://github.com/Vishal-V/GSoC/tree/master/face_app) -  [tensorflow/examples #83](https://github.com/tensorflow/examples/pull/83)
- [**Mask R-CNN**](https://github.com/Vishal-V/GSoC/tree/master/mask_rcnn) -  [tensorflow/examples #78](https://github.com/tensorflow/examples/pull/78)
- [**Mask R-CNN Notebook**](https://github.com/Vishal-V/GSoC/tree/master/mask_rcnn/notebooks) -  [tensorflow/examples #78](https://github.com/tensorflow/examples/pull/78)
- [**Hyperparameter Search**](https://github.com/Vishal-V/GSoC/blob/master/keras_tuner/hyperparamter_search.ipynb) -  [tensorflow/examples #84](https://github.com/tensorflow/examples/pull/84)
- [**Custom ResNet for TinyImageNet**](https://github.com/Vishal-V/GSoC/tree/master/tiny_imagenet_custom_resnet/model) -  [tensorflow/examples #79](https://github.com/tensorflow/examples/pull/79)
- [**Custom ResNet for TinyImageNet Notebook**](https://github.com/Vishal-V/GSoC/blob/master/tiny_imagenet_custom_resnet/tiny_imagenet_custom_resnet.ipynb) -  [tensorflow/examples #79](https://github.com/tensorflow/examples/pull/79)
## **Progress**
All pull requests can be found here [PRs Link](https://github.com/tensorflow/examples/pulls/Vishal-V)
- **Phase 1**: Completed the autoencoder migration and the corresponding notebook with both eager mode and graph mode training. Worked on the Boosted Trees Estimator and found a few bugs that caused the model to not learn at all. I then worked on the Keras-Tuner logo classifier based off of the Google I/O demo and also completed the R&D for the Custom ResNet model.
- **Phase 2**: Completed the R&D and created the entire StackGAN model for the CUB 200 birds dataset. Trained the model for 250 epochs on my GPU with decent results. Completed the basic Mask R-CNN migration and was left with some bugs and architectural changes necessary to be TF 2.0 compatible. I also completed training and testing 10-12 ResNet models across 5 colab accounts to get the most efficient strategy. 
- **Phase 3**: Completed the R&D and created the Age Conditional GAN for the Wiki cropped faces dataset. The plan was to train it on GCP and upload the weights for developers to make their own FaceApp, but due to the lack of cloud credits, this was put away for the time being. The final custom resnet model was chosen from among the multiple models built and the corresponding notebook was completed. The final autoencoder notebook was also completed. All models and their respecive bugs were fixed and the corresponding documentation was updated.
## **What's left?**
- Mask R-CNN fixes and possible rebuild from scratch to be TF 2.0 native.
- Age-cGAN Trained Weights for developers to build their own version of FaceApp.
- The Age-cGAN preprocesses all 62k images and stores it in a list just once to use it for every epoch without having to preprocess all the images over and over again. This has a significany overhead first up but eases it for the rest of the training. This may have a better solution.
## **Challenges**
- The most challenging task was the Mask R-CNN migration. I spent a lot of time on this and even translated the autodiff code that used graph functions. The migration was not fully completed and a possible rebuild from scratch might be a better alternative.
## **Learnings**
- I learnt a whole lot of best practices in these 3 months along with building for scale and writing maintainable code. I got an in-depth look into tensorflow as a framework and all the amazing code written that makes this the most popular AI framework. 
- Not to forget the amazing mentors I had who helped me along the way and solved all my doubts at the earliest. I also got a taste of just how much effort goes into benchmarking and creating official deep learning model implementations.
