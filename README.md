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
|[SIREN](https://arxiv.org/abs/2006.09661v1)|CVPR '20 |[Oficial Implementation](https://github.com/vsitzmann/siren) for Image Inpainting|

<!-- |[MUNIT](https://arxiv.org/abs/1804.04732)|ECCV '18 |[SoTA](https://github.com/nvlabs/MUNIT) for Cityscapes, Edge-to-Handbags [Image Translation]| -->
#
## **Progress**
|Evaluation|Task|Link|Status|Pull Request|
|---|---|---|---|---|
|E1| FineGAN Model |[Here](https://github.com/Vishal-V/tf-models/tree/master/finegan)| Complete |[ #9173](https://github.com/tensorflow/models/pull/9173), [ #8750](https://github.com/tensorflow/models/pull/8750)|
|E2| FineGAN Training Pipeline |[Here](https://github.com/Vishal-V/tf-models/tree/master/finegan)| Complete |[ #9173](https://github.com/tensorflow/models/pull/9173), [ #8750](https://github.com/tensorflow/models/pull/8750)|
|E3| FineGAN Benchmark |[Here](https://github.com/Vishal-V/tf-models/tree/master/finegan)| Partial |[ #9173](https://github.com/tensorflow/models/pull/9173)|
|E3| FineGAN - Notebook |[Here](https://github.com/Vishal-V/tf-models/blob/master/finegan/notebooks/efficient%20trials.ipynb)| Partial |[ #9173](https://github.com/tensorflow/models/pull/9173)|
|E3| Expressivity of BN - Code|[Here](https://github.com/Vishal-V/tf-models/tree/master/onlyBN)|Complete| [ #9174](https://github.com/tensorflow/models/pull/9174)|
|E3| Expressivity of BN - Notebook |[Here](https://github.com/Vishal-V/tf-models/tree/master/onlyBN/notebooks)|Complete| [ #9174](https://github.com/tensorflow/models/pull/9174)|
|E3| SIREN - Notebook|[Here](https://github.com/Vishal-V/tf-models/tree/master/siren/notebooks/)|Partial| [ ]()|
|E3| SIREN - Code|[Here](https://github.com/Vishal-V/tf-models/tree/master/siren/)|In Progress| [ ]()|
|E3| Detection Transformer |[Here](https://github.com/Vishal-V/tf-models/tree/master/detr)|In Progress| [ ]()|

<!-- |E3| MUNIT |[Here](https://github.com/Vishal-V/tf-models/munit/)|In Progress| [ ]()| -->

#
#
## **Work Done**
### FineGAN
Implemented the 3-Stage FineGAN Architecture with a stage each for Background, Foreground Outline and Foreground Mask generation. This includes the PatchGAN Background scene detector to detect background only patches for the auxiliary discriminator in the Background stage. I also fixed several errors and added workarounds for assigning values to eager tensors and assigning zero weight to the tensors with cropped real bounding boxes. To improve the data loading performance, I added the `tf.data` pipelines training and evaluation paired images with the data augmentation and the modified masked bounding boxes. Model created from the [paper](https://arxiv.org/abs/1811.11155) and compared with the author's code release to train on the CUB 200 dataset for birds. Model training provides results but is yet to be benchmarked [In Progress].
  
- **Model**: https://github.com/Vishal-V/tf-models/tree/master/finegan  
- **Notebook**: [GitHub Link](https://github.com/Vishal-V/tf-models/blob/master/finegan/notebooks/efficient%20trials.ipynb)
  
### Only BatchNorm
The expressive power of BatchNormalization is an under investigated topic. This [paper](https://arxiv.org/abs/2003.00152) from FAIR goes on to investigate the expressivity that comes from the 'beta' and 'gamma' parameters of BatchNormalization with extensive ablation studies and experiments. The particular position of 'gamma' and 'beta' as per-feature coefficient and bias. Batchnorm makes the optimization landscape smoother and also decouples the optimization of the weight magnitude and the direction of gradients. Like a gift that keeps on giving, BatchNorm also performs a novel regularization and explicitly casues the gradients to reach equilibrium. This model and the associated notebook revolves around reproducing the results from the paper. 
  
Two points to consider: The TensorFlow image translate function from TF Addons and did not perform as well as the paper authors have claimed. The same holds for the weight_decay that I used as SGDW and also as l2 kernel regularizer but in both cases, the training diverged. 
- **Model**: https://github.com/Vishal-V/tf-models/tree/master/onlyBN
- **Instructions**: https://github.com/Vishal-V/tf-models/tree/master/onlyBN/README.md
- **Notebook**: [GitHub Link](https://github.com/Vishal-V/tf-models/tree/master/onlyBN/notebooks/), [Colab Link](https://colab.research.google.com/drive/1Kr0zXT-OkyM51VDwUd1vBsFm8GLp8RTN?usp=sharing)
  
### OnlyBN ResNet Experiments
Running the OnlyBN ResNet model experiments (without frozen training) with the TinyImageNet dataset with progressive resizing and DepthwiseSeparableConv2D for efficient training. The model can also be used with all the layers except the BatchNorm layers set to be trainable. The author's benchmark of 32% for ImageNet with ResNet-200 may not be of much use to reproduce the results of, hence this notebook discusses alternative techniques including Cyclic Learning Rate.
- **Model**: https://github.com/Vishal-V/tf-models/tree/master/onlyBN/notebooks/efficient_tiny.ipynb
- **Instructions**: https://github.com/Vishal-V/tf-models/tree/master/onlyBN/README.md
- **Notebook**: [GitHub Link](https://github.com/Vishal-V/tf-models/tree/master/onlyBN/notebooks/efficient_tiny.ipynb), [Colab Link](https://colab.research.google.com/drive/1Kk1t_i2k9iO02rWO_uO39DawmBVEOffZ?usp=sharing)
    
### SIREN
Implicit Neural Representations with Periodic Activation Functions or SIREN is an exciting new paper from CVPR 2020 that can be used for a phenomenal array of applications including image inpainting, fitting an image, fitting an audio signal and even solving Poisson's equation! This notebook is a minimal demo of the model for fitting an image. The benchmarks for the same is in progress.
- **Model**: https://github.com/Vishal-V/tf-models/tree/master/siren/
- **Notebook**: [GitHub Link](https://github.com/Vishal-V/tf-models/tree/master/siren/notebooks/)

<!-- [Colab Link](https://colab.research.google.com/drive/15Mqrbtv5u39P9UOwIFjJWkpW6YsS7nA8) -->
  
### Detection Transformer (DETR)
End-to-end object detector with Transformer is a set based detector that uses the transformer architecture on top of a convolution backbone to perform tasks such as object detection and panoptic segmentation. The model takes in positional encodings for the transformer encoder. The set global loss that forces unique predictions for via bipartite matching and the encoder-decoder architecture does away with handcoded features and non-max suppression.
- **Model**: https://github.com/Vishal-V/tf-models/tree/master/detr
  
#
### TensorFlow User Group Community Page
Posted well researched and concise posts for the TFUG Community page from March 27 to Aug 28. I helped the page gain followers regularly and interacted with all impressions of the pots. In all, I helped grow the page from 2,750 followers to 31,325 followers over the course of the 3 month GSoC period.
- **Page**: https://www.linkedin.com/company/tensorflow
  
<!-- ### MUNIT (Multimodal Image-to-Image Translation)
MUNIT is one of the classic Image-to-Image translation models that is able to learn a multimodal distribution and model translations from various domains. MUNIT does this by decomposing the image style and content into their domains and then recombining them in diverse ways. The implementation although SoTA for Edge-to-Handbags, this implementation is more concerned towards the Cityscapes dataset for various weather domain translation. [In Progress]
- **Model**:  https://github.com/Vishal-V/tf-models/munit/
- **Instructions**: https://github.com/Vishal-V/tf-models/munit/README.md -->
  
#    
## **Progress**
<!-- All pull requests can be found here [PRs Link](https://github.com/tensorflow/examples/pulls/Vishal-V) -->
### **Phase 1**  
Implemented the 3-Stage FineGAN Architecture. This included the background generator, background auxiliary discriminator, background discriminator, parent generator, parent mask generator branch, parent information maximization discriminator, child stage generator, child mask generator branch and the child stage discriminators. I also got to interact with the paper authors. They were happy that their work was getting recognized in the community!

I used TensorFlow best practices for the implementation and ensured each part worked individually. The end-to-end training loop was next to come and this formed the deliverables for my first evaluation. After the evaluation, I also discussed the next models to work on with my mentor. I decided to work on DETR (Detection Transformer) along with the implementation of Transformer as completed by an intern.
 
### **Phase 2**  
I worked on integrating the individual FineGAN components into an effiicent training loop and met with several errors that TensorFlow did not support. I fixed several errors and added workarounds for assigning values to eager tensors and assigning zero weight to the tensors with cropped real bounding boxes. This led to a few efficiency problems as the graph mode training speedup could not be used while working with eager tensors. To improve the data loading performance, I added the tf.data pipelines training and evaluation paired images with the data augmentation and the modified masked bounding boxes. I was finally able to get the model working and the gradients passing correctly, but was unable to reproduce the author's results. So, FineGAN was a partial implementation at this stage and for this evaluation.
  
One amazing experience was the SpineNet tutorial and the implementation best-practices discussed by the TensorFlow team in a meeting along with the SpineNet paper authors from Google Brain. This was a very rare and interesting experience and got a whole lot out of attending it! I read the SpineNet paper and the DETR paper towards potentially starting to work on it and get a few components completed in time for the final evaluation and then continue working on it later.
  
### **Phase 3**  
Even with several changes and improvements added to the FineGAN model, the model didn't converge. So, I started comparing the model architecture as closely as possible to the author's codebase. I even performed model diagnosis to see if there are some spurious gradients being passed, but that was not the case. After training for several runs of 300 epochs (full training is for 600 epochs + hard negative mining) for the FineGAN model, I realized that either the model architecture is still missing some components or the workarounds I added to the code that PyTorch automatically allows might have made some difference. With the risk of having nothing to show for for the final evaluation, I started binge reading several of the newest papers in the internal paper list and from the latest conferences (CVPR 2020 and ECCV 2020) that I could even use for my full-time research work.
  
I got a fantastic learning experience from reading all those papers! My mentor Jaeyoun was very supportive during this phase and gave me the confidence to continue working on other models and get back to FineGAN later. I decided to work on Facebook's Only BN and Stanford's SIREN along with some components for DETR as a partial implementation. While the implementation for OnlyBN was smooth sailing, I had another blocker in the TensorFlow impelmentation of SGD with weight decay and the image translate augmentation function. Surprisingly, the ResNet-218, ResNet-434 and ResNet-866 diverged during the training scheme which forced me to tru other workarounds that still reached good enough performance and upheld the very essence of the investigations in the paper.
  
Although this is the last phase for GSoC 2020 and my last time as a GSoC student developer (I participated last year with the TensorFlow team!), I will continue working on impactful and interesting models for the TF team and the research community and benchmark SoTA models!
#
## **What's next?**
- FineGAN diagnosis and discussions with the authors to benchmark the model.
- Work on the DETR model architecture and complete the implementation for the new TF Model Garden repository with custom built-in components. Since the DETR benchmark is for Panoptic Segmentation, this work will also integrate well with the Panoptic FPN implementation that my GSoC colleague [@syiming](https://github.com/syiming) is working on for the TF Object Detection API.
- Work on other relevant models such as the DRIT++ and the MUNIT models for multimodal image-to-image translation and benchmark it for larger datastes.
  
## **Challenges**
- The most challenging task was coming up with improvements and workarounds to get the FineGAN model working and efficiently at that. I spent a lot of time on this and even translated the autodiff code that used graph functions. The benchmark reproducibility was not fully completed and a possible deeper diagnosis with help from the authors might help.
  
## **Learnings**
- I learnt a whole lot of best practices in these 3 months along with building for scale and writing maintainable code. I got an in-depth look into TensorFlow as a framework and all the amazing code written that makes this the most popular AI framework.
- I also got to attend one of the Google Brain tutorial meetings and got to know more about the vision for the TF Object Detection API and the new Computer Vision components being added to the TF Model Garden. 
- Not to forget the amazing mentorship from Jayeoun Kim who helped me out when I needed it the most. I also got a taste of just how much effort goes into benchmarking and creating official deep learning model implementations and learnt an important lesson in model reproducibility that even seasoned researchers may face from time-to-time! 
#
<!-- ## **Deliverables**
- [**Autoencoder**](https://github.com/Vishal-V/GSoC/tree/master/autoencoder/model) - [tensorflow/examples #68](https://github.com/tensorflow/examples/pull/68)
-  [**Autoencoder - Notebook**](https://github.com/Vishal-V/GSoC/tree/master/autoencoder/model) - [tensorflow/examples #68](https://github.com/tensorflow/examples/pull/68)
- [**StackGAN**](https://github.com/Vishal-V/GSoC/tree/master/stack_gan) -  [tensorflow/examples #77](https://github.com/tensorflow/examples/pull/77)
- [**Age-cGAN**](https://github.com/Vishal-V/GSoC/tree/master/face_app) -  [tensorflow/examples #83](https://github.com/tensorflow/examples/pull/83)
- [**Mask R-CNN**](https://github.com/Vishal-V/GSoC/tree/master/mask_rcnn) -  [tensorflow/examples #78](https://github.com/tensorflow/examples/pull/78)
- [**Mask R-CNN Notebook**](https://github.com/Vishal-V/GSoC/tree/master/mask_rcnn/notebooks) -  [tensorflow/examples #78](https://github.com/tensorflow/examples/pull/78)
- [**Hyperparameter Search**](https://github.com/Vishal-V/GSoC/blob/master/keras_tuner/hyperparamter_search.ipynb) -  [tensorflow/examples #84](https://github.com/tensorflow/examples/pull/84)
- [**Custom ResNet for TinyImageNet**](https://github.com/Vishal-V/GSoC/tree/master/tiny_imagenet_custom_resnet/model) -  [tensorflow/examples #79](https://github.com/tensorflow/examples/pull/79)
- [**Custom ResNet for TinyImageNet Notebook**](https://github.com/Vishal-V/GSoC/blob/master/tiny_imagenet_custom_resnet/tiny_imagenet_custom_resnet.ipynb) -  [tensorflow/examples #79](https://github.com/tensorflow/examples/pull/79) -->
