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

우선 **integer**(정수)를 8bit로 표현한 세 가지 예시를 살펴보자. 

![integer](images/integers.png)

- 첫 번째: unsigned integer

  range: $[0, 2^{n} - 1]$

- 두 번째: (signed integer) Sign-Magnitude

  range: $[-2^{n-1} - 1, 2^{n-1} - 1]$

   > 00000000과 10000000은 모두 0을 표현한다.

- 세 번째: (signed integer) 2-bit complement Representation

  range: $[-2^{n-1}, 2^{n-1} - 1]$

   > 00000000은 0, 10000000은 $-2^{n-1}$ 을 표현한다.

---

### 5.2.2 fixed-point number

소수(**decimal**)를 표현하는 방식은 두 가지가 있다.

- **fixed-point number**(고정 소수점 연산)

- **floating-point number**(부동 소수점 연산)

아래는 8bit fixed-point number를 나타낸 그림이다.

![fixed-point](images/fixed_point.png)

- 맨 앞 1bit는 sign bit로 사용한다.

- 3bits로 integer(정수)를 표현한다.

- 4bits로 fraction(소수)을 표현한다.

> 두 번째와 세 번째 연산의 차이: 소수점( $2^{-4}$ ) 의 위치를 나중에 곱하였다.

위와 같은 예시를 `fixed<w,b>`로 표현할 수 있다. `w`가 총 bit width, `b`가 fraction bit width이다.

> 32bit 예시: 1bit sign bit, 15bit integer, 16bit fraction

---

### 5.2.3 floating-point number

다음은 32bit **floating-point** number의 예시다.(가장 보편적인 **IEEE 754** 방법)

![32bit floating-point](images/32bit_floating_point.png)

$$ (-1)^{sign} \times (1 + \mathrm{Fraction}) \times 2^{\mathrm{Exponent} - 1} $$

- sign: 부호를 나타내는 1bit

- **exponent**: 지수를 나타내는 8bit

- fraction(mantissa): 가수를 나타내는 23bit

> 32bit(4byte)는 single precision(단정도), 64bit(8byte)는 double precision(배정도)이다.

### <span style='background-color: #393E46; color: #F7F7F7'>&nbsp;&nbsp;&nbsp;📝 예제 1: IEEE 754 표준에 따라 숫자 표현하기 &nbsp;&nbsp;&nbsp;</span>

숫자 -314.625를 IEEE 754 표준에 따라 표현하라.

### <span style='background-color: #C2B2B2; color: #F7F7F7'>&nbsp;&nbsp;&nbsp;🔍 풀이&nbsp;&nbsp;&nbsp;</span>

1. 음수이므로 **sign bit**는 1이다.

2. **fraction**

    -314.625의 절댓값 $314.625$ 를 2진수로 변환하면 ${100111010.101}_{(2)}$ 가 된다.

    - 소수점을 옮겨서 일의 자리 수, 소수점 형태로 만든다. 
    
    - 소수점 부분만을 fraction 23bit 부분에 맨 앞부터 채운다.

      > 남는 자리는  0으로 채운다.

$$ 1.00111010101 \times 2^{8} $$

3. **exponent**

    bias를 계산해야 한다. (bias = $2^{k-1}$ )
    
    - $k$ : exponent 부분의 bit 수를 나타낸다. 
    
    $$2^{8-1} = 127$$

    8 + 127 = 135를 2진수로 변환하면 ${10000111}_{(2)}$ 이 된다.

    - 변환한 2진수를 8bit exponent 부분에 채워준다.

결과는 다음과 같다.

| sign bit | exponent | fraction |
| :---: | :---: | :---: | 
| 1 | 10000111 | 00111010101000000000000 | 

---

### 5.2.4 floating-point number comparison

다양한 floating-point number 표현법을 비교해보자. 특히 neural network에서는 <U>fraction보다도 exponent에 더 민감</U>하기 떄문에, exponent 정보를 최대한 보존하는 표현법이 등장했다.

- underflow, overflow, NaN을 더 잘 처리하기 위해서는, exponent을 최대한 보존하여 정확도를 유지해야 한다.

- 더 작은 bit를 사용하면서 memory, latency는 줄이고, accuracy는 최대한 보존하는 것이 목표.

![floating point ex](images/floating_point_ex.png)

- **Half Precision**(FP16)

    exponent 5 bit, fraction은 10 bit

- Brain Float(BF16)

    IEEE FP32와 비교했을 때, exponent 7bit로 줄였지만 fraction은 8bit로 유지했다. 

- TensorFloat(TF32)
    
    exponent 10bit, fraction 8bit이다. 
    
    > FP16과 동일한 exponent(10bit), FP32와 동일한 fraction(8bit)를 지원한다.

    > BERT 모델에서 TF32 V100을 이용한 학습이, FP32 A100을 이용한 학습에 비해 6배 speedup을 달성했다.

---

## 5.3 Efficient Weights Quantization

그렇다면 quantization bits는 어느 정도가 효율적일까? 아래는 CNN에서 다양한 precision으로 quantization했을 때 정확도를 나타낸 도표다.

![quantization bits](images/quantization_bits.png)

- Conv layer: 4bits 이상

- FC layer: 2bits 이상

---

### 5.3.1 Huffman Coding

> [Huffman coding 정리](https://velog.io/@junhok82/%ED%97%88%ED%94%84%EB%A7%8C-%EC%BD%94%EB%94%A9Huffman-coding)

추가로 **Huffman Coding** 알고리즘을 적용하면 memory usage를 더 줄일 수 있다.

> Unix의 파일 압축, JPEG, MP3 압축에서 주로 사용된다.

우선 압축을 하는 원리를 살펴보자. 예시로 A, B, C라는 알파벳을 압축하여 표현할 것이다. 순전히 ASCII code로 표현하려고 한다면 8bits x 3으로 24bits를 사용해야 한다. 하지만 Huffman coding을 이용해 가변 길이의 code로 만들 것이다.

우선 a, b, c를 다음과 같이 압축하여 정의했다고 하자.

| a | b | c |
| :---: | :---: | :---: |
| 01 | 101 | 010 |

- a와 c의 접두어 부분이 겹친다.(`01`)

위처럼 시작 부분이 겹치는 경우 **prefix code**(접두어 코드) 방식으로 가변 코드를 만들 수 없다. 반면 아래 예시를 보자.

| a | b | c |
| :---: | :---: | :---: |
| 01 | 10 | 111 |

- 겹치는 접두어가 없다.

이 경우 `01 10 111` 총 7bits로 압축할 수 있다.

여기서 숫자를 결정짓는 것은 '문자의 빈도 수'이다. 빈도 수가 높은 문자일수록 짧은 길이의 code를 부여하고, 빈도 수가 낮은 문자일수록 긴 길이의 code를 부여한다.

이를 neural network에 적용하면 다음과 같다.

- 자주 나오는 weights: bit 수를 적게 사용해서 표현한다.

- 드문 weights: bit 수를 더 사용해서 표현한다.

대표적으로 [Deep Compression 논문](https://arxiv.org/pdf/1510.00149.pdf)에서는 'Pruning + K-Means-based quantization + Huffman Coding'을 적용하여 LeNet-5 모델에서 약 39배 Compression ratio를 달성했다.

![Deep Compression](images/deep_compression.png)

---

## 5.4 Neural Network Quantization

ImageNet dataset으로 훈련한 AlexNet에서 pruning+quantization, pruning, quantization 방법별 'accuracy와 compression ratio'를 비교해 보자.

![accuracy vs compression rate](images/acc_loss_and_model_compression.png)

- 가로: Compression Ratio, 세로: Accuracy loss

- 두 방법을 동시에 적용했을 때 accuracy의 보존율이 높다.

이제 neural network 도메인에서 다양한 quantization 방법을 살펴보자. 아래와 같은 floating-point number로 구성된 matrix를 quantization한다고 가정하자.

![floating-point matrix](images/floating-point_matrix.png)

- 저장: Floating-Point Weights

- 연산: Floating-Point Arithmetic

---

### 5.4.1 K-Means-based Quantization

**K-Means-based weight quantization**이란 여러 <U>bucket을 갖는 codebook</U>(**centroids**, 무게중심)을 만들어서 quantization하는 방식이다.

> 예를 들어 Computer Graphics에서는, 65536개의 스펙트럼으로 이루어진 원래 색상을 256개의 bucket을 갖는 codebook을 만들어서 quantization을 수행한다.

![K-Means-based_Quantization](images/K-Means-based_Quantization.png)

- 저장: **Integer** Weights, Floating-Point Codebook

- 연산: Floating-Point Arithmetic

> 예제에서 codebook의 cluster index는 0~3까지의 2bit로 표현된다.

### <span style='background-color: #393E46; color: #F7F7F7'>&nbsp;&nbsp;&nbsp;📝 예제 2: K-Means-based Quantization의 메모리 사용량 &nbsp;&nbsp;&nbsp;</span>

K-Means-based Quantization 이전/이후 필요한 memory를 계산하라.

### <span style='background-color: #C2B2B2; color: #F7F7F7'>&nbsp;&nbsp;&nbsp;🔍 풀이&nbsp;&nbsp;&nbsp;</span>

- before

    32bits floating point type 4x4 행렬의 weight를 저장한다.
    
  $$ 32 \times (4 \times 4) = 512 $$
    
    따라서 총 512bits = 64bytes이다.

- after

    행렬 내 값은 2bit cluster index를 갖는다.

    $$ 2 \times (4 \times 4) = 32$$
    
    따라서 행렬은 32 bits = 4 bytes를 갖는다.

    또한 codebook은 32bit floating point로 1x4 행렬을 갖는다.

    $$ 32 \times (1 \times 4) = 128$$

    따라서 codebook은 128bits = 16bytes를 갖는다.

    그러므로 quantization 이후 필요한 메모리 사용량은 20byte이다.
    
양자화 전후를 비교했을 때, 64/20=3.2로 약 3.2배 메모리 사용량이 감소했다.

> 예시보다 weight가 많은 행렬에서 더 큰 효과를 볼 수 있다.(약 32/N배 감소한다.)

---

#### 5.4.1.1 K-Means-based Quantization Error

위 양자화 예시에서 weight를 다시 reconstruct한 뒤, 기존과 비교하여 error를 계산해 보자.

![K-Means error](images/K-Means_error.png)

이처럼 quantization 시 필연적으로 error가 발생하게 된다. 하지만 추가로 centroids(codebook)을 fine-tuning하는 방식으로 error를 줄일 수 있다.

![Fine-tuning quantized weights(K-means)](images/K-means_fine_tune.png)

그러나 K-Means-based weight quantization은, weight만 integer type으로 바꾼 뒤, 실제 추론 상황에서는 다시 floating-point로 바꾸어야 한다는 단점이 있다.

> runtime inference 중 weight는 lookup table에 따라서 decompressed된다.(예제: 2bit int to 32bit float)

따라서 <U>오직 storage cost만 줄일 수 있다</U>는 한계를 지닌다. 실제 computation 과정, memory access에서는 여전히 floating-point를 사용한다.

---

### 5.4.2 Linear Quantization

이번에는 **Linear Quantization** 방법을 살펴보자. 마찬가지로 linear quantization도 codebook을 사용해서 quantized weights를 만들어낸다.

하지만 이때 **centroids**가 linear하다는 특징을 갖는다.(일정한 step size를 갖는다.)

![uniform quantization](images/uniform_quantization.png)

예시 weight 행렬에 linear quantization을 적용하는 과정을 살펴보자.

![linear quantization](images/linear_quantization.png)

> zero point, scale 계산법은 뒤에서 살필 것이다.

우선 위 그림의 나열 순서대로 수식을 표현하면 다음과 같다.

> FP weight, quantized weight, zero point, scale

$$ r = (q - Z) \times S $$

- $r$ : (floating-point) real number

- $q$ : (**integer**) quantized number

- $Z$ : (**integer**) zero point. 

  real number $r=0$ 에 정확히 mapping될 수 있도록 조절하는 역할이다. **offset**으로도 지칭한다.

- $S$ : (floating-point) scale

   scaling factor 역할이다.

이때 quantization하는 범위가 음의 정수를 포함하는가에 따라서 `unsigned int`, `signed int`를 사용할 수 있다. ReLU와 같이 음수 값을 포함하지 않는 경우에는 `unsigned int`를 주로 사용한다.

---

#### 5.4.2.1 zero point, scale 

이제 real number를 quantized number에 mapping하면서, quantization parameter인 zero point, scale을 계산해 보자.

수식은 기본적으로 최대, 최소 실수값을 가지고 계산한다.

> 주로 outlier(이상치)를 제거(**clipping**)한 범위의 최대, 최소값을 사용한다.

![linear quantization mapping](images/linear_quantization_mapping.png)

$$ r_{max} = S(q_{max} - Z) $$

$$ r_{min} = S(q_{min} - Z) $$

위 식을 정리하면 scaling factor에 대한 식을 얻을 수 있다.

$$ S = {{r_{max} - r_{min}} \over {q_{max} - q_{min}}} $$

### <span style='background-color: #393E46; color: #F7F7F7'>&nbsp;&nbsp;&nbsp;📝 예제 3: linear quantization &nbsp;&nbsp;&nbsp;</span>

예시 weight matrix에서 zero point, scale 값을 구하여라.

![floating-point matrix](images/floating-point_matrix.png)

### <span style='background-color: #C2B2B2; color: #F7F7F7'>&nbsp;&nbsp;&nbsp;🔍 풀이&nbsp;&nbsp;&nbsp;</span>

우선 weight 행렬에서 FP 최대값은 2.12, FP 최소값은 -1.08이다. 또한 integer은 각각 1과 -2에 대응된다. 이를 식에 대입하면 다음과 같이 scaling factor를 구할 수 있다.

$$ S = {{2.12 - (-1.08)} \over {1 - (-2)}} = 1.07 $$

$S$ 를 구했으므로 앞서 $r_{min}$ 혹은 $r_{max}$ 값에 대입하는 것으로 $Z$ 를 구할 수 있다. 

> 이때 $Z$ 가 정수가 되도록 round 연산을 적용해야 한다.

$$ Z = \mathrm{round}{\left( q_{min} - {{r_{min}} \over S} \right)} $$

값을 대입하면 다음과 같다.

$$ Z = \mathrm{round}{\left( -2 - {{-1.08} \over {1.07}} \right)} = 1 $$

따라서 zero point는 1이다.

---

#### 5.4.2.2 linear quantized matrix multiplication

이러한 linear quantization을 행렬 연산 관점에서, real number를 integer로 변환하는 **affine mapping**으로 볼 수 있다.

$$ Y = WX $$

$$ S_{Y}(q_{Y} - Z_{Y}) =  S_{W}(q_{W} - Z_{W}) \cdot S_{X}(q_{X} - Z_{X}) $$

이를 $q_{Y}$ 에 관한 식으로 정리하면 다음과 같다.

$$ q_{Y} = {{S_{W}S_{X}} \over {S_{Y}}}(q_{W} - Z_{W})(q_{X} - Z_{X}) + Z_{Y} $$

이 식을 전개하면 세 가지 항으로 연산을 나눠서 볼 수 있다.

$$ q_{Y} = {{S_{W}S_{X}} \over {S_{Y}}}(q_{W}q_{X} - Z_{W}q_{X} - Z_{X}q_{W} - Z_{W}Z_{X}) + Z_{Y} $$

- ${S_{W}S_{X}} \over {S_{Y}}$ : N-bit integer로 rescale한다. (0, 1) 범위의 값을 갖는다.

- $q_{W}q_{X} - Z_{W}q_{X} - Z_{X}q_{W} - Z_{W}Z_{X}$ : N-bit Integer multiplication과 32-bit Integer Addition/Subtraction 연산이다.

    - 여기서 $-Z_{W}q_{W} + Z_{W}Z_{X}$ 는 이미 계산하여 얻은 항이다.

- $Z_{Y}$ : N-bit Integer addition 연산이다.

---
