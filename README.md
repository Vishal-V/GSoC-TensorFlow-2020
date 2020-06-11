<img src="assets/gsoc_tf.png" width="600px" height="300px"/>
  
## Google Summer of Code 2020 - TensorFlow  
  
### **Mentors**
- Jaeyoun Kim ([@jaeyounkim](https://github.com/jaeyounkim))
- Yanhui Liang ([@yhliang2018](https://github.com/yhliang2018))
- Paige Bailey ([@dynamicwebpaige](https://github.com/dynamicwebpaige))  
  
## **Aim**
A collection of state-of-the-art Deep Learning models for the TensorFlow Model Garden implemented from their papers and trained on the datasets they are benchmarked on. My aim is to add some of the most exciting models to the Model Garden to help make model prototyping much faster and promote research using TF 2.x.
  
The project proposes models such as FineGan, Panoptic Segmentation and several other SoTA models. The TF Model Garden makes building new models with state-of-the-art performance much easier especially with the new additions to TF 2.x that have made every part of the model training pipeline much more efficient (TFRT is out!) The models I have proposed to add will be trained till they converge to the benchmarks in the paper and will have detailed documentation and lucid Colab demos to make them extensible and easy to use.  
# 
|Paper|Conference|Library Used|Benchmark|Priority|
|---|---|---|---|---|
|[FineGAN](https://arxiv.org/abs/1811.11155)|CVPR ‘19 |Python|SoTA for CUB 128x128, CUB, Stanford Cars [Image Generation]|High|
|[Panoptic-FPN](https://arxiv.org/abs/1901.02446)|CVPR ‘19 |Python|Ex-SoTA for COCO Panoptic [Detectron 2 Based Panoptic Segmentation]|High|
|[DetectoRS](https://arxiv.org/abs/2006.02334v1)|ArXiv|Python|[SoTA](https://github.com/joe-siyuan-qiao/DetectoRS) for COCO test-dev [Object Detection]|High|  
#
## **Progress**
|Evaluation|Task|Link|Status|Pull Request|
|---|---|---|---|---|
|E1| FineGan Implementation |[Here](https://github.com/Vishal-V/tf-models/tree/master/finegan)| WIP ||
|E2| Mask R-CNN |[Here](https://github.com/Vishal-V/tf-models/)|WIP|[ #78](https://github.com/tensorflow/examples/pull/78)|
|E2| Panoptic Segmentation |[Here](https://github.com/Vishal-V/tf-models/)|Research||
#