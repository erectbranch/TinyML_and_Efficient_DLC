<div width="100%" height="100%" align="center">
  
<h1 align="center">
  <p align="center">TinyML and Efficient Deep Learning Computing</p>
  <a href="https://www.youtube.com/playlist?list=PL80kAHvQbh-ocildRaxjjBy6MR1ZsNCU7">
  </a>
</h1>
  
  
<b>강의 주제: TinyML and Efficient Deep Learning Computing</b></br>
Instructor : Song Han(Associate Professor, MIT EECS)</br>
[[schedule](https://efficientml.ai/schedule/)] | [[youtube](https://www.youtube.com/playlist?list=PL80kAHvQbh-ocildRaxjjBy6MR1ZsNCU7)] | [[github](https://github.com/mit-han-lab/6s965-fall2022)]</b>

</div>

## :bulb: 목표

- **효율적인 추론 방법 공부**

  > DNN 연산에 있어서 효율성을 높일 수 있는 알고리즘을 공부한다.

- **제한된 성능에서의 DNN 모델 구성**

  > 디바이스의 제약에 맞춘 효율적인 DNN 모델을 구성한다.

</br>

## 🚩 정리한 문서 목록

### 📔 Efficient Inference

- [Pruning](https://github.com/erectbranch/TinyML_and_Efficient_DLC/tree/master/lec03)

  > unstructured/structured pruning
  
  > magnitude-based pruning(L1-norm), second-order-based pruning, percentage-of-zero-based pruning, regression-based pruning

- [Neural Architecture Search: basic concepts & manually-designed neural networks](https://github.com/erectbranch/TinyML_and_Efficient_DLC/tree/master/lec07/summary01)

  > input stem, stage, head
  
  > AlexNet, VGGNet, SqueezeNet(global average pooling, fire module, pointwise convolution), ResNet50(bottleneck block, residual learning), ResNeXt(grouped convolution)
  
  > MobileNet(depthwise-separable convolution), MobileNetV2(inverted bottleneck block), ShuffleNet(channel shuffle), SENet(squeeze-and-excitation block)

- [Neural Architecture Search: RNN controller & search strategy](https://github.com/erectbranch/TinyML_and_Efficient_DLC/tree/master/lec07/summary02)

  > RNN controller, cell-level search space, network-level search space

  > Search Strategy: grid search, random search, reinforcement learning, bayesian optimization, gradient-based search, evolutionary search

  > EfficientNet(compound scaling), ProxylessNAS(architecture parameter, mixed operation function, binary gate), DARTS(latency penalty term)

- [Neural Architecture Search: latency profiling & Once-for-All Network](https://github.com/erectbranch/TinyML_and_Efficient_DLC/tree/master/lec08)

  > MACs/FLOPs and latency, latency predictor(layer-wise latency profiling, network-wise latency profiling), specialized models for different hardware(mobile, CPU, GPU)

  > Once-for-All Network: progressive shrinking(elastic resolution/kernel size/depth/width)

- [Knowledge Distillation](https://github.com/erectbranch/TinyML_and_Efficient_DLC/tree/master/lec10/summary01)

  > NetAug, knowledge transfer(KT)
  
  > knowledge distillation: distillation loss, softmax temperature, matching intermediate weights/features/attention maps/sparsity pattern 

- [MCUNet](https://github.com/erectbranch/TinyML_and_Efficient_DLC/tree/master/lec11)

  > microcontroller, flash/SRAM usage, peak SRAM usage, MCUNet: TinyNAS, TinyEngine

  > TinyNAS: automated search space optimization(weight/resolution multiplier), resource-constrained model specialization(Once-for-All)

  > MCUNetV2: patch-based inference, network redistribution, joint automated search for optimization, MCUNetV2 architecture(VWW dataset inference)

  > RNNPool, MicroNets(MOPs & latency/energy consumption relationship)

</br>

## :mag: Schedule

### Lecture 1: Introduction

[ [slides](https://hanlab.mit.edu/files/course/slides/MIT-TinyML-Lec01-Introduction.pdf) ]

### Lecture 2: Basics of Deep Learning

[ [slides](https://hanlab.mit.edu/files/course/slides/MIT-TinyML-Lec02-Basics-of-Neural-Networks.pdf) | [video](https://youtu.be/5HpLyZd1h0Q) ]

---
## Efficient Inference
---

### Lecture 3: Pruning and Sparsity (Part I)

[ [slides](https://hanlab.mit.edu/files/course/slides/MIT-TinyML-Lec03-Pruning-I.pdf) | [video](https://youtu.be/sZzc6tAtTrM) ]

### Lecture 4: Pruning and Sparsity (Part II)

[ [slides](https://hanlab.mit.edu/files/course/slides/MIT-TinyML-Lec04-Pruning-II.pdf) | [video](https://youtu.be/1njtOcYNAmg) ]

### Lecture 5: Quantization (Part I)

[ [slides](https://hanlab.mit.edu/files/course/slides/MIT-TinyML-Lec05-Quantization-I.pdf) | [video](https://youtu.be/91stHPsxwig) ]

### Lecture 6: Quantization (Part II)

[ [slides](https://hanlab.mit.edu/files/course/slides/MIT-TinyML-Lec06-Quantization-II.pdf) | [video](https://youtu.be/sYpl97ToNdg) ]

### Lecture 7: Neural Architecture Search 
(Part I)

[ [slides](https://hanlab.mit.edu/files/course/slides/MIT-TinyML-Lec07-NAS-I.pdf) | [video](https://youtu.be/NQj5TkqX48Q) ]

### Lecture 8: Neural Architecture Search
(Part II)

[ [slides](https://hanlab.mit.edu/files/course/slides/MIT-TinyML-Lec08-NAS-II.pdf) | [video](https://youtu.be/UlvkBZdOhpg) ]

### Lecture 9: Neural Architecture Search
(Part III)

[ [slides](https://hanlab.mit.edu/files/course/slides/MIT-TinyML-Lec09-NAS-III.pdf) | [video](https://youtu.be/_cvn9pflblk) ]

### Lecture 10: Knowledge Distillation

[ [slides](https://hanlab.mit.edu/files/course/slides/MIT-TinyML-Lec10-Knowledge-Distillation.pdf) | [video](https://youtu.be/IIqf-oUTHe0) ]

### Lecture 11: MCUNet - Tiny Neural Network 
Design for Microcontrollers

[ [slides](https://hanlab.mit.edu/files/course/slides/MIT-TinyML-Lec11-MCUNet.pdf) | [video](https://youtu.be/Hi4I0ZtPsbY) ]

~~Lecture 12: Paper Reading Presentation~~

---

## Efficient Training and System Support

---

### Lecture 13: Distributed Training and Gradient Compression (Part I)

[ [slides](https://hanlab.mit.edu/files/course/slides/MIT-TinyML-Lec13-Distributed-Training-I.pdf) | [video](https://youtu.be/oIIy6nmMoeM) ]

### Lecture 14: Distributed Training and Gradient Compression (Part II)

[ [slides](https://hanlab.mit.edu/files/course/slides/MIT-TinyML-Lec14-Distributed-Training-II.pdf) | [video](https://youtu.be/7W0MCjc8OD4) ]

### Lecture 15: On-Device Training and Transfer Learning (Part I)

[ [slides](https://hanlab.mit.edu/files/course/slides/MIT-TinyML-Lec15-On-Device-Training-And-Transfer-Learning-I.pdf) | [video](https://youtu.be/P_tVABpgb6w) ]

### Lecture 16: On-Device Training and Transfer Learning (Part II)

[ [slides](https://hanlab.mit.edu/files/course/slides/MIT-TinyML-Lec16-On-Device-Training-And-Transfer-Learning-II.pdf) | [video](https://youtu.be/rG-KM8eVzj8) ]

### Lecture 17: TinyEngine - Efficient Training and Inference on Microcontrollers

[ [slides](https://hanlab.mit.edu/files/course/slides/MIT-TinyML-Lec17-TinyEngine.pdf) | [video](https://youtu.be/oCMnJXH0c50) ]

---

## Application-Specific Optimizations

---

### Lecture 18: Efficient Point Cloud Recognition

[ [slides](https://hanlab.mit.edu/files/course/slides/MIT-TinyML-Lec18-Efficient-Point-Cloud-Recognition.pdf) | [video](https://youtu.be/fKIxpM-F0zw) ]

### Lecture 19: Efficient Video Understanding and GANs

[ [slides](https://hanlab.mit.edu/files/course/slides/MIT-TinyML-Lec19-Efficient-Video-Understanding-GANs.pdf) | [video](https://youtu.be/J4olmnIwgtk) ]

### Lecture 20: Efficient Transformers

[ [slides](https://hanlab.mit.edu/files/course/slides/MIT-TinyML-Lec20-Efficient-Transformers.pdf) | [video](https://youtu.be/RGUCmX1fvOE) ]

---

## Quantum ML

---

### Lecture 21: Basics of Quantum Computing

[ [slides](https://hanlab.mit.edu/files/course/slides/MIT-TinyML-Lec21-Quantum-Basics.pdf) | [video](https://youtu.be/8eT1QTVb1uo) ]

### Lecture 22: Quantum Machine Learning

[ [slides](https://hanlab.mit.edu/files/course/slides/MIT-TinyML-Lec22-Quantum-ML.pdf) | [video](https://youtu.be/20ftuhSV4sk) ]

### Lecture 23: Noise Robust Quantum ML

[ [slides](https://hanlab.mit.edu/files/course/slides/MIT-TinyML-Lec23-Noise-Robust-Quantum-ML.pdf) | [video](https://youtu.be/1gV0u8SfXe8) ]

~~Lecture 24: Final Project Presentation~~

~~Lecture 25: Final Project Presentation~~

### Lecture 26: Course Summary & Guest Lecture

[ [slides](https://hanlab.mit.edu/files/course/slides/MIT-TinyML-Lec25-AIMET.pdf) | [video](https://youtu.be/NCuLGvCeYl8) ]
