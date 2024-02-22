<div width="100%" height="100%" align="center">
  
<h1 align="center">
  <p align="center">TinyML and Efficient Deep Learning Computing</p>
  <a href="https://www.youtube.com/playlist?list=PL80kAHvQbh-ocildRaxjjBy6MR1ZsNCU7">
  </a>
</h1>
  
  
<b>강의 주제: TinyML and Efficient Deep Learning Computing</b></br>
Instructor : Song Han(Associate Professor, MIT EECS)</br>
Fall 2023([[schedule](https://hanlab.mit.edu/courses/2023-fall-65940)] | [[youtube](https://youtube.com/playlist?list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB&si=TmPWEvwUR79TVxrz)]) | Fall 2022([[schedule](https://hanlab.mit.edu/courses/2022-fall-6s965)] | [[youtube](https://www.youtube.com/playlist?list=PL80kAHvQbh-ocildRaxjjBy6MR1ZsNCU7)])</b>

</div>

## :bulb: 목표

- **효율적인 추론 방법 공부**

  > 딥러닝 연산에 있어서 효율성을 높일 수 있는 알고리즘을 공부한다.

- **제한된 성능에서의 딥러닝 모델 구성**

  > 디바이스의 제약에 맞춘 효율적인 딥러닝 모델을 구성한다.

</br>

## 🚩 정리한 문서 목록

### 📖 Basics of Deep Learning

- [Efficiency Metrics](https://github.com/erectbranch/TinyML_and_Efficient_DLC/tree/master/lec02/summary02)

  > latency, storage, energy

  > Memory-Related(\#parameters, model size, \#activations), Computation(MACs, FLOP)

### 📔 Efficient Inference

- [Pruning Granularity, Pruning Critertion](https://github.com/erectbranch/TinyML_and_Efficient_DLC/tree/master/lec03)

  > Unstructured/Structured pruning(Fine-grained/Pattern-based/Vector-level/Kernel-level/Channel-level)
  
  > Pruning Criterion: Magnitude(L1-norm, L2-norm), Sensitivity and Saliency(SNIP), Loss Change(First-Order, Second-Order Taylor Expansion)
  
  > Data-Aware Pruning Criterion: Average Percentage of Zero(APoZ), Reconstruction Error, Entropy

- [Automatic Pruning, Lottery Ticket Hypothesis](https://github.com/erectbranch/TinyML_and_Efficient_DLC/tree/master/lec04/summary01)

  > Finding Pruning Ratio: Reinforcement Learning based, Rule based, Regularization based, Meta-Learning based 

  > Lottery Ticket Hypothesis(Winning Ticket, Iterative Magnitude Pruning, Scaling Limitation)

  > Pruning at Initialization(Connection Sensitivity, Gradient Flow)

- [System & Hardware Support for Sparsity](https://github.com/erectbranch/TinyML_and_Efficient_DLC/tree/master/lec04/summary02)

  > EIE(CSC format: relative index, column pointer)

  > M:N Sparsity

  ---

- [Basic Concepts of Quantization](https://github.com/erectbranch/TinyML_and_Efficient_DLC/tree/master/lec05/summary01)

  > Numeric Data Types: Integer, Fixed-Point, Floating-Point(IEEE FP32/FP16, BF16, NVIDIA FP8), INT4 and FP4

  > Uniform vs Non-uniform quantization, Symmetric vs Asymmetric quantization

- [Vector Quantization, Linear Quantization](https://github.com/erectbranch/TinyML_and_Efficient_DLC/tree/master/lec05/summary02)

  > Vector Quantization(VQ): Deep Compression(iterative pruning, retrain codebook, Huffman encoding), Product Quantization(PQ): AND THE BIT GOES DOWN
  
  > Linear Quantization: Zero point, Scaling Factor, Quantization Error(clip error, round error), Linear Quantized Matrix Multiplization(FC layer, Conv layer)

- [Post Training Quantization](https://github.com/erectbranch/TinyML_and_Efficient_DLC/tree/master/lec06/summary01)

  > Weight Quantiztion: Per-Tensor Activation Per-Channel Activation, Group Quantization(Per-Vector, MX), Weight Equalization, Adative Rounding

  > Activation Quantization: During training(EMA), Calibration(Min-Max, KL-divergence, Mean Squared Error)

  > Bias Correction, Zero-Shot Quantization(ZeroQ)

- [Quantization-Aware Training, Low bit-width quantization](https://github.com/erectbranch/TinyML_and_Efficient_DLC/tree/master/lec06/summary02)

  > Fake quantization, Straight-Through Estimator

  > Binary Quantization(Deterministic, Stochastic, XNOR-Net), Ternary Quantization

  ---

- [Neural Architecture Search: basic concepts & manually-designed neural networks](https://github.com/erectbranch/TinyML_and_Efficient_DLC/tree/master/lec07/summary01)

  > input stem, stage, head
  
  > AlexNet, VGGNet, SqueezeNet(global average pooling, fire module, pointwise convolution), ResNet50(bottleneck block, residual learning), ResNeXt(grouped convolution)
  
  > MobileNet(depthwise-separable convolution, width/resolution multiplier), MobileNetV2(inverted bottleneck block), ShuffleNet(channel shuffle), SENet(squeeze-and-excitation block), MobileNetV3(redesigning expensive layers, h-swish)

- [Neural Architecture Search: Search Space](https://github.com/erectbranch/TinyML_and_Efficient_DLC/tree/master/lec07/summary02)

  > Search Space: Macro, Chain-Structured, Cell-based(NASNet), Hierarchical(Auto-DeepLab, NAS-FPN)

  > design search space: Cumulative Error Distribution, FLOPs distribution, zero-cost proxy

- [Neural Architecture Search: Performance Estimation & Hardware-Aware NAS](https://github.com/erectbranch/TinyML_and_Efficient_DLC/tree/master/lec08)

  > Weight Inheritance, HyperNetwork, Weight Sharing(super-network, sub-network)

  > Performance Estimation Heuristics: Zen-NAS, GradSign

  ---

- [Knowledge Distillation](https://github.com/erectbranch/TinyML_and_Efficient_DLC/tree/master/lec10/summary01)

  > Knowledge Distillation(distillation loss, softmax temperature)
  
  > What to Match?: intermediate weights, features(attention maps), sparsity pattern, relational information

  > Distillation Scheme: Offline Distillation, Online Distillation, Self-Distillation

- [Distillation for Applications](https://github.com/erectbranch/TinyML_and_Efficient_DLC/tree/master/lec10/summary02)

  > Applications: Object Detection, Semantic Segmentation, GAN, NLP

  > Tiny Neural Network: NetAug

  ---

- [MCUNet](https://github.com/erectbranch/TinyML_and_Efficient_DLC/tree/master/lec11)

  > Microcontroller, MCUNet(TinyNAS, TinyEngine), automated search space optimization(weight/resolution multiplier), resource-constrained model specialization(Once-for-All)

  > MCUNetV2: patch-based inference, network redistribution, joint automated search for optimization, MCUNetV2 architecture(VWW dataset inference)

### ⚙️ Efficient Training and System Support

- [TinyEngine](https://github.com/erectbranch/TinyML_and_Efficient_DLC/tree/master/lec17)

  > memory hierarchy of MCU, data layout(NCHW, NHWC, CHWN)

  > TinyEngine: Loop Unrolling, Loop Reordering, Loop Tiling, SIMD programming, Im2col, In-place depthwise convolution, appropriate data layout(pointwise, depthwise convolution), Winograd convolution

  ---

### 🔧 Application-Specific Optimizations

- [Efficient Video(live) Understanding](https://github.com/erectbranch/TinyML_and_Efficient_DLC/tree/master/lec19/summary01)

  > 2D CNNs for Video(live) Understanding, 3D CNNs for Video(live) Understanding(I3D), Temporal Shift Module(TSM)

  > Other Efficient Methods: Kernel Decomposition, Multi-Scale Modeling, Neural Architecture Search(X3D), Skipping Redundant Frames/Clips, Utilizing Spatial Redundancy

- [Generative Adversarial Networks (GANs)](https://github.com/erectbranch/TinyML_and_Efficient_DLC/tree/master/lec19/summary02)

  > GANs(Generator, Discriminator), Conditional/Unconditional GANs, Difficulties in GANs

  > Compress Generator(GAN Compression), Dynamic Cost GANs(Anycost GANs), Data-Efficient GANs(Differentiable Augmenatation)

  ---

- [Transformer](https://github.com/erectbranch/TinyML_and_Efficient_DLC/tree/master/2023-lec12/summary01)

  > NLP Task(Discriminative, Generative), Pre-Transformer Era(RNN, LSTM, CNN)

  > Transformer: Tokenizer, Embedding, Multi-Head Attention, Feed-Forward Network, Layer Normalization(Pre-Norm, Post-Norm), Positional Encoding

- [Transformer Design Variants](https://github.com/erectbranch/TinyML_and_Efficient_DLC/tree/master/2023-lec12/summary02)

  > Encoder-Decoder(T5), Encoder-only(BERT), Decoder-only(GPT), Relative Positional Encoding, KV cache optimization, Gated Linear Unit

- [Efficient Vision Transformer](https://github.com/erectbranch/TinyML_and_Efficient_DLC/tree/master/2023-lec14/summary01)

  > Vision Transformer, Window Attention(Swin Transformer), Sparse Window Attention(FlatFormer), ReLU Linear Attention(EfficientViT), Sparsity-Aware Adaptation(SparseViT)

</br>

## :mag: Schedule (6.S965 • Fall 2022)

| Date | Lecture | Youtube | Slide |
| --- | --- | --- | --- |
| **Sep 8** | Lecture 1: **Introduction** | - | [[slides](https://www.dropbox.com/scl/fi/ym2frcworou622a7wrghb/lec01.pdf?rlkey=nbnjyn0wyyvhmoti7jvqbfa9s&dl=0)] |
| **Sep 13** | Lecture 2: **Basics of Deep Learning** | [[video](https://youtu.be/5HpLyZd1h0Q)] | [[slides](https://www.dropbox.com/scl/fi/7q21t5meajse0mapbtice/lec02.pdf?rlkey=x1ls1ktyv8taomitdkhl7etqq&dl=0)] |
| | | | |
| | **Efficient Inference** | | |
| | | | |
| **Sep 15** | Lecture 3: **Pruning and Sparsity (Part I)** | [[video](https://youtu.be/sZzc6tAtTrM)] | [[slides](https://www.dropbox.com/scl/fi/vns8vgzfrjjqrjovqtrxw/lec03.pdf?rlkey=nwofk3suges17224m7idg9nwm&dl=0)] |
| **Sep 20** | Lecture 4: **Pruning and Sparsity (Part II)** | [[video](https://youtu.be/1njtOcYNAmg)][[video(live)](https://youtu.be/fWP3Q6tNtYU)] | [[slides](https://www.dropbox.com/scl/fi/3ghge1dxv1lu74mnj5ktv/lec04.pdf?rlkey=afa5so7ybdu60o4zbkstr80pd&dl=0)] |
| **Sep 22** | Lecture 5: **Quantization (Part I)** | [[video](https://youtu.be/91stHPsxwig)][[video(live)](https://youtu.be/AlASZb93rrc)] | [[slides](https://www.dropbox.com/scl/fi/i4wmuy1wgs0urtqezzr7w/lec05.pdf?rlkey=iqme56qke3zb5g5cq7zxmvmhn&dl=0)]  |
| **Sep 27** | Lecture 6: **Quantization (Part II)** | [[video](https://youtu.be/sYpl97ToNdg)][[video(live)](https://hanlab.mit.edu/courses/2022-fall-6s965#)] | [[slides](https://www.dropbox.com/scl/fi/j8x1g500rvkd1pbjw84m3/lec06.pdf?rlkey=d9i4d2k0q6nu38lp2r7nlaw4b&dl=0)] |
| **Sep 29** | Lecture 7: **Neural Architecture Search (Part I)** | [[video](https://youtu.be/NQj5TkqX48Q)][[video(live)](https://youtu.be/NQj5TkqX48Q)] | [[slides](https://www.dropbox.com/scl/fi/fnzeebhd9227pzr0opgol/lec07.pdf?rlkey=c612kexszsjbd3jkubyzb4jah&dl=0)] |
| **Oct 4** | Lecture 8: **Neural Architecture Search (Part II)** | [[video](https://youtu.be/UlvkBZdOhpg)][[video(live)](https://youtu.be/PFitZnPIKoc)] | [[slides](https://www.dropbox.com/scl/fi/mnamxcqlnglm35q1y20wz/lec08.pdf?rlkey=q3ckspu8o7vauuloh3ostlotc&dl=0)] |
| **Oct 6** | Lecture 9: **Neural Architecture Search (Part III)** | [[video](https://youtu.be/_cvn9pflblk)][[video(live)](https://youtu.be/_cvn9pflblk)] | [[slides](https://www.dropbox.com/scl/fi/t5dhodsg2wqkn1rgsf1lx/lec09.pdf?rlkey=xzz8vrwlwjc6vm8foinnbecpv&dl=0)] |
| **Oct 13** | Lecture 10: **Knowledge Distillation** | [[video](https://youtu.be/IIqf-oUTHe0)][[video(live)](https://youtu.be/tT9Lnt6stwA)] | [[slides](https://www.dropbox.com/scl/fi/7x4i8bf3ush5bdt0mu57k/lec10.pdf?rlkey=7viyngsy60imiilpkxqbmva1l&dl=0)] |
| **Oct 18** | Lecture 11: **MCUNet - Tiny Neural Network Design for Microcontrollers** | [[video](https://youtu.be/Hi4I0ZtPsbY)][[video(live)](https://youtu.be/YBER-SNlkqs)] | [[slides](https://www.dropbox.com/scl/fi/1b9ozxzrzk8x3lwh4lc0b/lec11.pdf?rlkey=g4wbzq8h88l9dnl94svsnn8fs&dl=0)] |
| | | | |
| | **Efficient Training and System Support** |  | |
| | | | |
| **Oct 25** | Lecture 13: **Distributed Training and Gradient Compression (Part I)** | [[video](https://youtu.be/oIIy6nmMoeM)][[video(live)](https://youtu.be/WYY1nbTWAk4)] | [[slides](https://www.dropbox.com/scl/fi/qv7luhv8v0qnj94jpafnt/lec13.pdf?rlkey=djzxg2wbglrthzdtamr8e0zv1&dl=0)] |
| **Oct 27** | Lecture 14: **Distributed Training and Gradient Compression (Part II)** | [[video](https://youtu.be/7W0MCjc8OD4)][[video(live)](https://youtu.be/2VdmlWxY1fE)] | [[slides](https://www.dropbox.com/scl/fi/tmpdgvy6we0vrj9s9q3a4/lec14.pdf?rlkey=lmi6y79y5w5hy67arx647jhh5&dl=0)] |
| **Nov 1** | Lecture 15: **On-Device Training and Transfer Learning (Part I)** | [[video](https://youtu.be/P_tVABpgb6w)][[video(live)](https://youtu.be/VW_6V0k_i30)] | [[slides](https://www.dropbox.com/scl/fi/l26nz3e8yyh4friogs9yf/lec15.pdf?rlkey=wbgijgrbgdfkjs27j34t5mvrl&dl=0)] |
| **Nov 3** | Lecture 16: **On-Device Training and Transfer Learning (Part II)** | [[video](https://youtu.be/rG-KM8eVzj8)][[video(live)](https://youtu.be/h_55fEBf6Fs)] | [[slides](https://www.dropbox.com/scl/fi/cbmnpivl6hibs5mj2tf50/lec16.pdf?rlkey=j3i0op9yd9vdfbxgt3gvrcd9u&dl=0)] |
| **Nov 8** | Lecture 17: **TinyEngine - Efficient Training and Inference on Microcontrollers** | [[video](https://youtu.be/oCMnJXH0c50)][[video(live)](https://youtu.be/usPxjVC7pr0)] | [[slides](https://www.dropbox.com/scl/fi/qioqdqdszys02v1upp25p/lec17.pdf?rlkey=yuffskefckc8oab0ntyi64on5&dl=0)] |
| | | | |
| | **Application-Specific Optimizations** |  | |
| | | | |
| **Nov 10** | Lecture 18: **Efficient Point Cloud Recognition** | [[video](https://youtu.be/fKIxpM-F0zw)][[video(live)](https://youtu.be/xtxRKbd_2W0)] | [[slides](https://www.dropbox.com/scl/fi/os1kyo9eixyvezajpt5tw/lec18.pdf?rlkey=d8cvpn6tjivu98fhnxdu0yl04&dl=0)] |
| **Nov 15** | Lecture 19: **Efficient Video(live) Understanding and GANs** | [[video](https://youtu.be/J4olmnIwgtk)][[video(live)](https://youtu.be/0WZSzStMgLk)] | [[slides](https://www.dropbox.com/scl/fi/a17vu1ujgxvuemev43k42/lec19.pdf?rlkey=1c9ljnyjdrtueljkp7xd4gunf&dl=0)] |
| **Nov 17** | Lecture 20: **Efficient Transformers** | [[video](https://youtu.be/RGUCmX1fvOE)][[video(live)](https://youtu.be/UYaJKavtCbU)] | [[slides](https://www.dropbox.com/scl/fi/ohk1emmjcfga0roafm2rk/lec20.pdf?rlkey=far16xdpurr900vzdrizg17mw&dl=0)] |
| | | | |
| | **Quantum ML** |  | |
| | | | |
| **Nov 22** | Lecture 21: **Basics of Quantum Computing** | [[video](https://youtu.be/8eT1QTVb1uo)][[video(live)](https://youtu.be/evTGcFnLu1g)] | [[slides](https://www.dropbox.com/scl/fi/emcqog86lsp18ku5fza0p/lec21.pdf?rlkey=co0nbj1wovxwvvvtlqogl9x15&dl=0)] |
| **Nov 29** | Lecture 22: **Quantum Machine Learning** | [[video](https://youtu.be/20ftuhSV4sk)][[video(live)](https://youtu.be/20ftuhSV4sk)] | [[slides](https://www.dropbox.com/scl/fi/x71s8rrffjrcnrpdc50w0/lec22.pdf?rlkey=9d14pbcey0do8nsac6uvu72x2&dl=0)] |
| **Dec 1** | Lecture 23: **Noise Robust Quantum ML** | [[video](https://youtu.be/1gV0u8SfXe8)][[video(live)](https://youtu.be/1gV0u8SfXe8)] | [[slides](https://www.dropbox.com/scl/fi/gllgdkus2717mrqjg8rab/lec23.pdf?rlkey=u6pqg98ppvlmfs5t6ev6l3wma&dl=0)] |
| **Dec 13** | Lecture 26: **Course Summary & Guest Lecture** | [[video](https://youtu.be/NCuLGvCeYl8)] | [[slides](https://www.dropbox.com/scl/fi/3svdkacflj9hv2yupsmpt/lec25.pdf?rlkey=r3q0kofbu06fw7ud0ihqboy0k&dl=0)] |
| | | | |