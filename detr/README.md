# DETR TF-2  
  
[![Paper](http://img.shields.io/badge/paper-arXiv.1811.11155-B3181B.svg)](https://arxiv.org/abs/1811.11155) 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vishal-V/tf-models/blob/master/...)  

This repository is the unofficial implementation of the following [[Paper]](https://arxiv.org/abs/2005.12872v3).

* End-to-End Object Detection with Transformers 

## Description

End-to-end object detector with Transformer is a set based detector that uses the transformer architecture on top of a convolution backbone to perform tasks such as object detection and panoptic segmentation. The model takes in positional encodings for the transformer encoder. The set global loss that forces unique predictions for via bipartite matching and the encoder-decoder architecture does away with handcoded features and non-max suppression.
  
<img src="../assets/detr.png">  
  
<!-- ## History

> :memo: Provide a changelog. -->
  
## Key Features

- [x] TensorFlow 2.3.0
- [ ] Inference example (Colab Demo)
- [ ] Transfer learning example
- [ ] Eager mode training with `tf.GradientTape` [If Required]
- [ ] Graph mode training with `model.train_on_batch`
- [x] Functional model with `tf.keras.layers`
- [ ] Input pipeline using `tf.data` and `tfds`
- [ ] Tensorflow Serving
- [ ] Vectorized transformations
- [x] GPU accelerated
- [ ] Fully integrated with `absl-py` from [abseil.io](https://abseil.io)
- [ ] Clean implementation
- [ ] Following the best practices
- [x] Apache 2.0 License

## Requirements

[![TensorFlow 2.3](https://img.shields.io/badge/tensorflow-2.3-brightgreen)](https://github.com/tensorflow/tensorflow/releases/tag/v2.2.0)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-382/)

> :memo: Provide details of the software required.  
>  
> * Add a `requirements.txt` file to the root directory for installing the necessary dependencies.  
>   * Describe how to install requirements using pip.  
> * Alternatively, create INSTALL.md.  

To install requirements:

```setup
pip install -r requirements.txt
```

## Results
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/end-to-end-object-detection-with-transformers/panoptic-segmentation-on-coco-panoptic)](https://paperswithcode.com/sota/panoptic-segmentation-on-coco-panoptic?p=end-to-end-object-detection-with-transformers)
  
### Panoptic Segmentation
 
| Model name | Download | PQ | AP |  
|------------|----------|----------------|----------------|  
| DETR-R101 | [Checkpoint](https://drive.google.com/...), [SavedModel](https://tfhub.dev/...) | xx% | xx% |  

## Dataset
  
COCO, COCO Mini, COCO Panoptic  
  
## Training

> :memo: Provide training information.  
>  
> * Provide details for preprocessing, hyperparameters, random seeds, and environment.  
> * Provide a command line example for training.  

Please run this command line for training.

```shell
python3 ...
```

## Evaluation

> :memo: Provide an evaluation script with details of how to reproduce results.  
>  
> * Describe data preprocessing / postprocessing steps.  
> * Provide a command line example for evaluation.  

Please run this command line for evaluation.

```shell
python3 ...
```

## References

> :memo: Provide links to references.  

## Citation

> :memo: Make your repository citable.  
>  
> * Reference: [Making Your Code Citable](https://guides.github.com/activities/citable-code/)  

If you want to cite this repository in your research paper, please use the following information.

## Authors or Maintainers

* Vishal Vinod ([@Vishal-V](https://github.com/Vishal-V))
  
This project is licensed under the terms of the **Apache License 2.0**.
