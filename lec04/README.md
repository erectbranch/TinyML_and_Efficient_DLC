# Lecture 04 - Pruning and Sparsity (Part II)

> [Lecture 04 - Pruning and Sparsity (Part II) | MIT 6.S965](https://youtu.be/1njtOcYNAmg)

---

## 4.1 Pruning Ratio

> [AMC: AutoML for Model Compression and Acceleration on Mobile Devices 논문(2018)](https://arxiv.org/abs/1802.03494)

per-layer **pruning ratio**는 어떻게 설정해야 할까? 우선 uniform shrink와 not uniform shrink를 복습하며 비교해 보자.

- FC layer일수록 pruning하기 쉬웠다.

- 얕은 레이어일수록 pruning하기 어려웠다.

![uniform vs not uniform](images/uniform_vs_not_uniform.png)

위 두 방식의 latency, accuracy를 비교한 그래프를 보자. 각 채널마다 적절한 pruning ratio를 찾아내면, latency와 accuracy를 모두 향상시킬 수 있다.

![AMC tradeoff](images/AMC_tradeoff.png)

- not uniform pruning이 uniform scaling과 비교했을 때 **Pareto Dominant**하다.

---

### 4.1.1 Finding Pruning Ratios

pruning ratio를 정하기 위해서는, 우선 해당 레이어가 pruning에 얼마나 **sensitive**한지 알아야 한다.

- sensitive한 레이어라면 pruning을 하면 할수록 정확도가 크게 감소한다.

- 따라서 레이어의 sensitivity를 알기 위한 **sensitivity analysis**가 필요하다.

다음은 CIFAR-10 데이터셋을 쓰는 VGG-11 model에서, heuristic한 방법으로 pruning ratio를 찾는 과정이다.

![VGG-11 pruning](images/VGG11_pruning.png)

- $L_i$ : $i$ 번째 레이어

   pruning ratio $r \in \lbrace 0, 0.1, 0.2, ..., 0.9  \rbrace$ 를 골라서 pruning을 적용한다.

결과를 보면 degrade ${\triangle} {Acc}_{r}^{i}$ 가 제일 큰 레이어는 $L0$ 이다. 즉 $L0$ 레이어가 제일 pruning에 민감하다.

위 그래프에 degradation threshold $T$ 를 추가하면, 각 레이어마다 어느 정도의 pruning ratio를 적용할지 알 수 있다.

![VGG-11 pruning threshold](images/VGG11_pruning_threshold.png)

하지만 이렇게 얻은 pruning ratio가 optimal하지는 않다. 각 레이어 성격과 레이어 사이의 interection을 고려하지 않았기 때문이다.(sub-optimal, 차선의 방식)

> 예를 들어 레이어가 pruning에 민감하지는 않지만, 해당 레이어 크기가 작다면 아무리 pruning ratio를 크게 설정해도 효과가 적을 것이다.

---

### 4.1.2 Automatic Pruning: AMC

그렇다면 레이어 사이의 interection을 고려하면서 pruning ratio를 찾을 수는 없을까? **AMC**(AutoML for Model Compression) 논문에서는 pruning ratio를 찾는 문제를 **reinforcement learning problem**(강화 학습 문제)으로 정의하여 해결한다.

![AMC](images/AMC.png)

- critic

    좋은 정책인지 나쁜 정책인지를 평가하기 위한 reward function이 필요하다. 
    
    - 가장 단순한 reward function로 -Error(error rate)를 사용할 수 있다.

    - 추가로 긴 latency나 큰 model size(FLOPs)를 고려해서 패널티로 부여할 수 있다.

      > latency constraints는 latency lookup table(LUT)을 미리 만들어서 구현한다.

```math
R = \begin{cases} -Error, & if \, satisfies \, constrains \\ -\infty , & if \, not \end{cases}
```

- actor

    - **Action**: 각 레이어가 갖는 **sparsity ratio**.(pruning ratio)

```math
a \in [0,1)
```

- Embedding

    강화 학습을 위해서는 network architecture를 embedding해야 한다.

    - $s$ : **state**. 11개 feature를 갖는다.

        layer index $i$ , channel number, kernel sizes, FLOPs, ...


```math
s_t = [N, C, H, W, i]
```

- Agent

    DDPG agent를 사용한다.(continuous action output을 지원한다.)

다음은 강화 학습 문제를 해결함으로써 얻어낸 pruning policy(레이어별 sparsity ratio 분포)이다.(ImageNet 데이터셋으로 학습한 ResNet-50. 이름처럼 50개 레이어를 갖는다.) 그림을 보면 peak와 crest가 경향성을 갖는다.

![AMC sparsity ratio](images/AMC_sparsity_ratio.png)

- y축: density(\#non-zero weight/ \#total weight)

    따라서 y축의 값이 작을수록 \#non-zero weight가 적으므로 sparsity가 크다. 

- **peaks**: 1x1 convolution은 pruning을 잘 적용하지 않았다.

- **crests**: 반면 3x3 convolution에서 더 aggressive하게 pruning된 것을 알 수 있다.

---

### 4.1.3 NetAdapt

> [NetAdapt: Platform-Aware Neural Network Adaptation for Mobile Applications 논문(2018)](https://arxiv.org/abs/1804.03230)

**NetAdapt**도 마찬가지로 레이어별 최적의 pruning ratio를 찾는 논문이다.

![NetAdapt](images/NetAdapt.png)

1. 각 iteration마다 수동으로 정의한 $\triangle R$ 만큼 latency가 줄어드는 것을 목표로 pruning을 지속한다.(latency LUT 기반)

    > \#models = \#iterations

2. pruning을 진행한 뒤, 각 레이어 $L_k$ (A~Z)마다 fine-tuning 후 정확도를 측정한다.(short-term fine-tune)

3. 가장 큰 정확도를 갖는 pruned layer를 선택한다.

    이후 accuracy를 회복하기 위한 long-term fine-tune을 진행한다.

---

## 4.2 Finetuning Pruned Neural Network

pruning 후 정확도를 회복하는 fine-tuning 과정에서는, 이미 해당 모델이 수렴에 가깝기 때문에 learning rate를 더 작게 설정해야 한다.

> 보통 original learning rate의 1/100이나 1/10으로 설정한다.

또한 pruning과 fine-tuning을 단계적으로 반복하며 수행하는 **iterative pruning**이 일반적인 pruning+fine-tuning 방법보다 효과적이다.

> 30% pruning + fine-tuning이 끝난 뒤, 40% pruning + fine-tuning, ...

![iterative pruning](images/iterative_pruning.png)

---

### 4.2.1 Regularization

fine-tuning 중 loss term에 **regularization** 항을 추가하는 것으로, weight를 더 sparse하게 만들 수 있다.

- penalize non-zero parameters

- encourage small parameters

가장 대표적인 regularization 방법은 **L1 Regularization**와 **L2 Regularization**이다.(양쪽 다 비슷한 비율로 자주 사용된다.)

- L1 Regularization

    - $L$: data loss

    - $\lambda$: regularization strength

```math
L' = L(x; W) + \lambda |W|
```

- L2 Regularization

```math
L' = L(x; W) + \lambda ||W||^2
```

---

## 4.3 Lottery Ticket Hypothesis

> [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks 논문](https://arxiv.org/abs/1803.03635)

그렇다면 sparse neural network를 from scratch부터 다시 학습하면 어떨까?(가중치를 무작위로 초기화한 뒤 다시 학습) 이러한 의문에 대한 답을 간단히 요약하면 다음과 같다.

- pruning을 통해 얻은 architecture는, 다시 학습하면 더 낮은 정확도를 얻을 가능성이 크다.

- 하지만 찾기는 어려워도 기존의 dense model보다 적은 패러미터를 가지면서, 더 적은 iteration 학습만으로도 동일한 성능을 갖는 sub-network(=**winning ticket**)가 존재할 가능성이 있다.

다음은 The Lottery Ticket Hypothesis 논문에서 weight를 비율에 따라 pruning한 뒤 모델을 다시 학습한 결과이다.

![Lottery Ticket Hypothesis graph](images/lottery_ticket_hypothesis_graph.png)

- 왼쪽: MNIST, 오른쪽: CIFAR-10 데이터셋

- 굉장히 aggressive한 pruning ratio로도(오직 weight를 1%만 남겨도) 정확도 회복이 가능하다.

---

### 4.3.1 Iterative Magnitude Pruning

이러한 winning ticket 모델은 **Iterative Magnitude Pruning** 방법을 통해 찾아낼 수 있다.

1. dense model 학습 $\rightarrow$ pruning $\rightarrow$ 동일한 sparsity pattern을 가지며 다른 weight를 갖는 모델이 되도록 무작위 초기화

    > 이를 **sparsity mask**라고 한다.

    ![iterative magnitude pruning 1](https://github.com/erectbranch/TinyML_and_Efficient_DLC/blob/master/lec04/images/iterative_magnitude_pruning_1.png)

2. 해당 모델을 다시 재학습 $\rightarrow$ pruning

    ![iterative magnitude pruning 2](https://github.com/erectbranch/TinyML_and_Efficient_DLC/blob/master/lec04/images/iterative_magnitude_pruning_2.png)

3. 2번을 통해 얻은 모델을 동일한 sparsity pattern을 갖는 무작위 가중치 모델로 초기화

    ![iterative magnitude pruning 3](https://github.com/erectbranch/TinyML_and_Efficient_DLC/blob/master/lec04/images/iterative_magnitude_pruning_3.png)

4. 2번과 3번 과정을 반복하며 winning ticket를 찾는다.

단, 이러한 Iterative Magnitude Pruning 방법은 수렴할 때까지 계속 학습해야 한다는 단점을 갖는다.

---

### 4.3.2 Scaling Limitation

> [Stabilizing the Lottery Ticket Hypothesis 논문](https://arxiv.org/abs/1903.01611)

단, MNIST, CIFAR-10과 같이 작은 데이터셋과 달리, ImageNet과 같이 거대한 데이터셋에서는 from scratch( $W_{t=0}$ )부터 학습해서 정확도가 복구되지 않는다. 

- 대신 그보다 적은 $k$ 번 횟수만큼 먼저 training iteration을 거친 뒤의 가중치( $W_{t=k}$ )를 사용하면, pruned sub-networks의 정확도를 회복할 수 있다.

  ![scaling limitation](https://github.com/erectbranch/TinyML_and_Efficient_DLC/blob/master/lec04/images/lottery_imagenet.png)

---


## 4.4 System Support for Sparsity

추론 시 input activation과 weight 사이의 연산을 수행하게 된다. 다음 세 가지 경우를 살펴보자.

- 0 \* A = 0

  weight가 0이므로 계산할 필요가 없다.(**weight sparsity**)

  - computation 10배 감소, memory footprint 5배 감소

- W \* 0 = 0

  activation이 0이므로 계산할 필요가 없다.(**activation sparsity**)

  > ReLU activation function으로, activation sparsity가 쉽게 얻어지게 된다.(activation sparsity를 조절하기 위해 ReLU를 개조하기도 한다.)

  - computation 3배 감소

- 2.09, 1.92 $\rightarrow$ 2
  
  2.09와 1.92가 모두 2라는 동일한 값을 가리키도록 하면, weight를 훨씬 줄일 수 있다. (**weight sharing**)

  > 높은 정밀도가 필요하지 않으므로 2로 근사할 수 있다.(quantization) 

  - memory footprint 8배 감소

---

### 4.4.1 EIE: Efficient Inference Engine

> [EIE: Efficient Inference Engine on Compressed Deep Neural Network 논문(2016)](https://arxiv.org/abs/1602.01528)

추론 중 다음과 같은 연산을 수행한다고 하자.

$$ \bar{a} = [0 \quad a_1 \quad 0 \quad a_3] $$

```math
\begin{bmatrix} w_{0,0} && w_{0,1} && 0 && w_{0,3} \\ 0 && 0 && w_{1,2} && 0 \\ 0 && w_{2,1} && 0 && w_{2,3} \\ 0 && 0 && 0 && 0 \\ 0 && 0 && w_{4,2} && w_{4,3} \\ w_{5,0} && 0 && 0 && 0 \\ 0 && 0 && 0 && w_{6,3} \\ 0 && w_{7,1} && 0 && 0 \end{bmatrix} \begin{bmatrix} b_0 \\ b_1 \\ -b_2 \\ b_3 \\ -b_4 \\ b_5 \\ b_6 \\ -b_7 \end{bmatrix} \underset{ReLU}{\Rightarrow} \begin{bmatrix} b_0 \\ b_1 \\ 0 \\ b_3 \\ 0 \\ b_5 \\ b_6 \\ 0 \end{bmatrix}
```

먼저 위 예제를 4개의 processing element로 나눠보자.

![processing element ex](images/pe_ex_1.png)

EIE 논문에서는 초록색 processing element를 메모리에 저장할 때, sparsity를 활용하여 물리적으로 다음과 같이 mapping한다.(오직 non-zero만 저장)

- **Relative Index**

  Absolute Index 대신 Relative Index를 사용하며 메모리 사용량을 줄인다.(bits 수도 감소)

  - $w_{0,0}$ : index 0

  - $w_{0,1}$ : $w_{0,0}$ 아래 0, $w_{0,0}$ 의 바로 옆에 위치하므로, index 1

  - $w_{4,2}$ : $w_{0,1}$ 아래 0, 오른쪽 0 다음에 위치하므로, index 2

  - $w_{0,3}$ , $w_{4,3}$ : 중간에 0 없이 바로 위치하므로 index 0 

- Column Pointer

  - starting point $w_{0,0}$ 를 기준으로 몇 번째 column에 위치하는지 정보를 나타낸다. 

| virtual weight | $w_{0,0}$ | $w_{0,1}$ | $w_{4,2}$ | $w_{0,3}$ | $w_{4,3}$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Relative Index | 0 | 1 | 2 | 0 | 0 |
| Column Pointer | 0 | 1 | 2 | 3 | 5 |

> 이러한 format을 **CSC format**(Compressed Sparse Column)으로 지칭한다.

---

#### 4.4.1.1 Dataflow

- activation 0

  skip

  ![dataflow 1](https://github.com/erectbranch/TinyML_and_Efficient_DLC/blob/master/lec04/images/dataflow_1.png)

- activation $a_1$

  broadcast 후 parallel하게 연산한다.

  ![dataflow 2-1](https://github.com/erectbranch/TinyML_and_Efficient_DLC/blob/master/lec04/images/dataflow_2_1.png)

  ![dataflow 2-2](https://github.com/erectbranch/TinyML_and_Efficient_DLC/blob/master/lec04/images/dataflow_2_2.png)

- activation 0

  ![dataflow 3](https://github.com/erectbranch/TinyML_and_Efficient_DLC/blob/master/lec04/images/dataflow_3.png)

- activation $a_3$

  broadcast 후 parallel하게 연산한다.

  ![dataflow 4-1](https://github.com/erectbranch/TinyML_and_Efficient_DLC/blob/master/lec04/images/dataflow_4_1.png)

  ![dataflow 4-2](https://github.com/erectbranch/TinyML_and_Efficient_DLC/blob/master/lec04/images/dataflow_4_2.png)

끝나면 다음 cycle로 가중치 행렬의 아래 부분 연산도 수행한다. 결과를 모두 얻은 뒤 update를 수행한다.

---

#### 4.4.1.2 Micro Architecture for each PE

각 PE는 다음과 같은 과정을 거쳐서 연산을 가속하여 수행하게 된다.

![PE](images/PE.png)

흐름도를 단계별로 파악해 보자.

- **Activation Sparsity**: zero/non-zero인지 검사한다.

  non-zero일 경우, weight와 연산하게 된다.

  ![PE graph 1](images/pe_graph_1.png)

- **Weight Sparsity**: non-zero weight 위치 정보를 register에 전달한다.

  ![PE graph 2](images/pe_graph_2.png)

- **ALU**: weight decoding 수행 및 address를 계산한다.

  > quantization과 관련된 자세한 내용은 lec05 참조

  - quantized weight를 decoding한다.

  - relative index 정보를 바탕으로 absolute address를 계산한다.(address accumulate)

  ![PE graph 3](images/pe_graph_3.png)

- **ALU**: 연산을 수행한다.

  activation과 decoded weight의 연산을 수행한다.

  ![PE graph 4](images/pe_graph_4.png)

- **Write Back**: SRAM에 output을 저장한다.

  ALU로 계산한 absolute address에 연산 결과를 저장한다.

  ![PE graph 5](images/pe_graph_5.png)

- **ReLU**, **Non-zero Detection**

  ReLU 연산을 수행하고, Non-zero 값을 찾아낸다.(next stage에서 활용)

  ![PE graph 6](images/pe_graph_6.png)

---