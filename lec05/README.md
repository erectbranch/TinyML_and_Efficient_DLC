# Lecture 05 - Quantization (Part I)

> [Lecture 05 - Quantization (Part I) | MIT 6.S965](https://youtu.be/91stHPsxwig)

---

## 5.1 Quantization

![quantized signal](images/quantized_signal.png)

continuous 혹은 large set of values 특성을 가진 input을 discrete set으로 변환하는 것을 **quantization**(양자화)라고 지칭한다.

![quantized image](images/quantized_image.png)

neural network에 quantization을 적용하기 전/후의 weight 분포 차이를 살펴보자.

![continuous weight](images/continuous-weight.png)

![discrete weight](images/discrete-weight.png)

> fine-tuning을 적용하면 여기서 조금 변화가 생긴다.

---

## 5.2 Numeric Data Types

---

### 5.2.1 Integer

우선 **integer**(정수)를 8bit로 표현하는 방법을 살펴보자. 

![integer](images/integers.png)

- 첫 번째: unsigned integer

    - range: $[0, 2^{n} - 1]$

- 두 번째: (signed integer) Sign-Magnitude

    - range: $[-2^{n-1} - 1, 2^{n-1} - 1]$

    > 00000000과 10000000은 모두 0을 표현한다.

- 세 번째: (signed integer) Two-bit complement Representation

    - range: $[-2^{n-1}, 2^{n-1} - 1]$

    > 00000000은 0, 10000000은 $-2^{n-1}$ 을 표현한다.

---

### 5.2.2 fixed-point number

이를 fixed-point number(고정 소수점 연산), floating-point number(부동 소수점 연산)과 비교하면 차이가 극명해 진다.

아래는 fixed-point number를 8bit로 표현한 그림이다.

![fixed-point](images/fixed_point.png)

- 두 번째와 세 번째 연산의 차이는, 소수점( $2^{-4}$ ) 의 위치를 나중에 곱해준 부분이다.

---

### 5.2.3 floating-point number

다음은 32bit **floating-point** number를 표현한 그림을 보자.(가장 보편적인 **IEEE 754** 방법)

> 32bit(4byte)는 single precision(단정도), 64bit(8byte)는 double precision(배정도)이다.

![32bit floating-point](images/32bit_floating_point.png)

$$ (-1)^{sign} \times (1 + \mathrm{Fraction}) \times 2^{\mathrm{Exponent} - 1} $$

- sign: 부호를 나타내는 1bit

- exponent: 지수를 나타내는 8bit

- fraction(mantissa): 가수를 나타내는 23bit

예를 들어 숫자 -314.625를 IEEE 754 표준에 따라 표현하면 다음과 같다.

1. 음수이므로 sign bit = 1

2. fraction

    - -314.625의 절댓값 $314.625$ 를 2진수로 변환하면 ${100111010.101}_{(2)}$ 가 된다.

    - 2진수의 소수점을 옮겨서 일의 자리 수와 소수점으로 표현되게 만든다. 그리고 소수점 부분을 fraction 23bit 부분에 맨 앞부터 채워준다.

    > 남는 자리는  0으로 채운다.

$$ 1.00111010101 \times 2^{8} $$

3. exponent

    - bias를 계산해야 한다. bias = $2^{k-1}$ 로 k는 exponent의 bit 수이다. 즉, 현재는 $2^{8-1} = 127$ 이 된다.

    - 8 + 127 = 135를 2진수로 변환하면 ${10000111}_{(2)}$ 이 된다.

    - 변환한 2진수를 8bit exponent 부분에 채워준다.

    > 결국 소수점에 관한 정보를 exponent에 담는 것이다.

이번에는 여러 floating-point number 표현법을 비교해보자. neural network는 <U>fraction보다도 exponent에 더 민감</U>하다는 점에 유의하자. 따라서 underflow, overflow, NaN을 더 잘 처리하기 위해서는 exponent을 최대한 보존해서 정확도를 유지하는 것이 중요하다.

> 더 작은 bit를 사용하면서 memory와 latency는 줄이며 accuracy는 보존하는 것이 목표이다.

![floating point ex](images/floating_point_ex.png)

- Half Precision(FP16)

- Brain Float(BF16): IEEE 754와 비교하면 Fraction은 7bit로 작지만, Exponent는 8bit로 동일하다. 

- TensorFloat(TF32): Fraction은 10, Exponent는 8bit이다. 이는 FP16와 동일한 수치 범위를 지원하면서 정밀도를 높인 버전이다.

---

## 5.3 Neural Network Quantization

그렇다면 과연 어느 정도의 bit수를 갖도록 quantization하는 것이 효율적일까?

![quantization bits](images/quantization_bits.png)

- 일반적으로 Conv layer에서는 4bits, FC layer에서는 2bits를 이상을 사용해야 한다.

여기에 **Huffman Coding**을 적용한다면, 여기서 더 memory usage를 줄이면서 quantization을 적용할 수도 있다.

- 자주 나오는 weights: bit 수를 적게 사용해서 표현한다.

- 드문 weights: bit 수를 더 사용해서 표현한다.

참고로 Deep Compression 논문에서는 'Pruning + K-Means-based quantization + Huffman Coding'을 적용하여 LeNet-5에서 약 39배 Compression ratio를 달성했다.

> [Deep Compression 논문](https://arxiv.org/pdf/1510.00149.pdf)

![Deep Compression](images/deep_compression.png)

이제 다양한 quantization 예시를 보자. 아래와 같은 floating-point number로 구성된 matrix가 있을 때, 이를 quantization하는 다양한 방법을 알아보자.

![floating-point matrix](images/floating-point_matrix.png)

> 32bit float의 weight라고 하자.

- storage: Floating-Point Weights

- Computation: Floating-Point Arithmetic

---

### 5.3.1 K-Means-based Quantization

**K-Means-based weight quantization**은 여러 bucket을 갖는 codebook을 만들어서 quantization을 수행한다.

> 예를 들어 Computer Graphics에서는, 65536개의 스펙트럼으로 이루어진 원래 색상을 256개의 bucket을 갖는 codebook을 만들어서 quantization을 수행한다.

![K-Means-based_Quantization](images/K-Means-based_Quantization.png)

- storage: Integer Weights, Floating-Point Codebook

- Compute: Floating-Point Arithmetic

행렬에 담긴 2bit cluster index와 centroids(codebook)로 구성된 것이 특징이다. quantization 이전/이후의 필요한 memory를 비교해 보자.

- before: floating point로 값을 저장했으면, 32bit \* (4*4)로 총 512bit(64byte)가 필요했다.

- after: 2bit \* (4*4) = 32bit(4byte)에 32bit \* 4 = 128bit(16byte), 즉 20byte만 있으면 된다.(약 3.2배 작아졌다.)

> 예시 행렬보다 weight가 많은 행렬에서 더 큰 효과를 볼 수 있다.(약 32/N배 작아진다.)

weight들을 다시 reconstruct한 뒤 error를 살펴보면 다음과 같다.

![K-Means error](images/K-Means_error.png)

여기서 추가로 centroids(codebook)을 fine-tuning하는 것도 가능하다.

![Fine-tuning quantized weights(K-means)](images/K-means_fine_tune.png)

이번에는 ImageNet dataset을 사용하는 AlexNet에서 quantization을 했을 때의 'accuracy와 compression ratio'를 비교해 보자.

![accuracy vs compression rate](images/acc_loss_and_model_compression.png)

- 가로 축은 Compression Ratio, 세로 축은 Accuracy loss를 의미한다.

- 왼쪽으로 갈수록 Quantization, Pruning이 더 많이 적용된 것이다.

neural network에서 K-Means-based weight quantization을 적용하면 다음과 같은 layer 순서로 진행된다.

![K-Means-based Weight Quantization](images/K-Means-based_Quantization.png)

- In Storage: storage에 저장되어 있었던 quantized weights

- During Computation: runtime inference 중 weight들은 lookup table에 따라서 decompressed된다.(2bit int to 32bit float)

하지만 이러한 K-Means-based weight quantization 방법은 <U>오직 storage cost만 줄일 수 있다</U>는 한계를 지닌다. 실제 computation과 memory access는 여전히 floating-point를 사용한다.

---

### 5.3.2 Linear Quantization

이번에는 **Linear Quantization** 방법을 살펴보자. linear quantization 역시 codebook을 사용해서 quantized weights를 만들어낸다. 이때 **centroids**가 linear한 특징을 갖는다.(스탭 크기가 일정하다.)

![linear quantization](images/linear_quantization.png)

> zero point, scale 계산법은 추후 살필 것이다.

위 그림의 식은 다음과 같이 나타낼 수 있다.

$$ r = (q - Z) \times S $$

- $r$ : (floating-point) real number

- $q$ : (integer) quantized number

- $Z$ : (integer) zero point. quantization parameter로, real number $r=0$ 에 정확히 mapping될 수 있도록 조절된다.

- $S$ : (floating-point) scale(scaling factor)

real number와 quantized number가 mapping되는 방식을 보면 scale이 어떻게 적용되는지 알 수 있다.

![linear quantization mapping](images/linear_quantization_mapping.png)

방금 식을 다시 살펴보면 real number의 max, min 값도 알 수 있다.

$$ r_{max} = S(q_{max} - Z) $$

$$ r_{min} = S(q_{min} - Z) $$

이 둘의 값을 이용해서 scale을 계산할 수 있다.

$$ S = {{r_{max} - r_{min}} \over {q_{max} - q_{min}}} $$

예를 들어 앞서 본 weight matrix를 예시로 scale 값을 계산하면 다음과 같다.

$$ S = {{2.12 - (-1.08)} \over {1 - (-2)}} = 1.07 $$

S를 구했으므로 앞서 $r_{min}$ 혹은 $r_{max}$ 값에 대입하는 것으로 $Z$ 를 구할 수 있다. 이때 정수가 되도록 round 연산을 적용하는 것에 유의해야 한다.

$$ Z = \mathrm{round}{\left( q_{min} - {{r_{min}} \over S} \right)} $$

예제에 식을 적용하면 다음과 같다.

$$ Z = \mathrm{round}{\left( -2 - {{-1.08} \over {1.07}} \right)} = 1 $$

---

#### 5.3.1.1 linear quantized matrix multiplication

이러한 linear quantization은 real number를 integer로 변환하는 **affine mapping**으로 볼 수 있다. 행렬 연산의 관점에서 이를 살펴보자.

$$ Y = WX $$

$$ S_{Y}(q_{Y} - Z_{Y}) =  S_{W}(q_{W} - Z_{W}) \cdot S_{X}(q_{X} - Z_{X}) $$

이를 $q_{Y}$ 에 관한 식으로 정리하면 다음과 같다.

$$ q_{Y} = {{S_{W}S_{X}} \over {S_{Y}}}(q_{W} - Z_{W})(q_{X} - Z_{X}) + Z_{Y} $$

이 식을 전개하면 세 가지 항으로 연산을 나눠서 볼 수 있다.

$$ q_{Y} = {{S_{W}S_{X}} \over {S_{Y}}}(q_{W}q_{X} - Z_{W}q_{X} - Z_{X}q_{W} - Z_{W}Z_{X}) + Z_{Y} $$

- ${S_{W}S_{X}} \over {S_{Y}}$ : N-bit integer로 rescale한다. (0, 1) 범위의 값을 갖는다.

- $q_{W}q_{X} - Z_{W}q_{X} - Z_{X}q_{W} - Z_{W}Z_{X}$ : N-bit Integer multiplication과 32-bit Integer Addition/Subtraction 연산이다.

    - 여기서 $-Z_{W}q_{W} + Z_{W}Z_{X}$ 는 precompute된 항이다.

- $Z_{Y}$ : N-bit Integer addition 연산이다.

---
