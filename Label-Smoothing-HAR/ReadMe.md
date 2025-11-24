# Label Smoothing Experiment

reference) `When Does Label Smoothing Help? (2020)`

# 1. 참조 논문 요약

- 수많은 DL에서 Label smoothing을 많이 사용하고 있지만 해당 논문이 나오기 전까진 왜 좋은건지 정확히 분석한 논문이 없었음.
- Label smoothing이 무엇인지는 설명 생략
- Logit이란? 특정 class에 대한 예측값을 softmax에 통과하기 전의 출력값
- 왜 효과적인가?
    - Loss Function으로 CE Loss를 사용한다고 가정.
    
    $
    L=-\sum\limits_{i=1}^K y_i\log p_i 
    $
    
    - Hard targets([1, 0, 0, 0])을 사용하면 = 정답 class를 제외하고 나머지는 y=0이 된다. → 해당 Loss Function을 통해 역전파를 수행할 때 **거의, 정답 class에 대해서만 집중적으로 학습**한다.
    
    - Label Smoothing([0.97, 0.1, 0.1, 0.1])을 사용하면 = **정답이 아닌 class에 대해서도 적절히 학습할 수 있다. → Over-confidence 경향을 줄일 수 있다.**

# 2. 실험 계획

- 4개 HAR dataset(UCI-HAR, WISDM, PAMAP2, mHEALTH)을 대상으로 Label Smoothing의 효과 탐구
- 모델: 간단한 1D ResNet 구조 & ModernTCN
- 비교 조건: 1) w/o Label Smoothing, 2) + Label Smoothing
- 평가 지표: F1, Acc, NLL, ECE, mean-max-SoftMax, t-SNE, feature intra/inter distance, entropy/margin, seperability ratio

# 3. 실험에 사용할 평가 지표

## 1) Mean Max SoftMax = “모델이 얼마나 자신하는지”

- Test set sample 하나에 대해:
    - 모델 출력 = softmax를 통과한 ‘각 클래스별 예측 확률’ → 이 중 가장 큰 값을 모든 sample에 대해서 평균 내면…
    
    $$
    mean \ max \ SoftMax=\frac {1}{N} \sum\limits_{i=1}^N max p_{i, k} 
    $$
    
- 값이 1에 가까울수록 ‘Confidence’가 높다.
- 논문에서 주장한대로 label smoothing이 confidence를 얼마나 감소시켜주는지 비교하는 용도

## 2) NNL (Negative Log-Likelihood) = “정답에 준 확률을 얼마나 크게 줬는지”

(CE Loss = NNL)

- 한 sample에 대해 예측 확률분포가 $p = softmax(logits)$ 일 때,

$$
NLL = -logp_y
$$

- 이를 모든 sample에 대해서 평균내서 $NLL_{avg}$를 구한다.
- $NLL_{avg}$는 확률분포의 분산이 클수록 커진다.

---

### 해석 방법)

**CASE #1** 

```python
샘플 1: p_y = 0.99  
샘플 2: p_y = 0.96  
샘플 3: p_y = 0.95  
샘플 4: p_y = 0.40  
샘플 5: p_y = 0.05  <-- 위험한 tail
```

- 분산이 높고 tail이 길다.
- **평균 NLL이 크게 나온다.**

→ “over-confident”

**CASE #2**

```python
샘플 1: p_y = 0.95  
샘플 2: p_y = 0.92  
샘플 3: p_y = 0.90  
샘플 4: p_y = 0.85  
샘플 5: p_y = 0.70  <-- tail 사라짐
```

- 평균은 조금 낮아졌지만 분산이 크게 감소했다.
- **평균 NLL이 작게 나온다.**

→ “less confident”

여기서, 과도한 confidence는 작은 변화에도 $p_y$가 급격히 감소하는 불안정성 야기. 

→ 즉, over-confident Model은

- train data와 비슷한 분포에선 NLL이 매우 작지만 distribution이 조금만 틀어져도 NLL이 극단적으로 커진다.

→ 다시말해, 분산이 큰 모델은 NLL이 급격히 커질 수 있다. (한쪽으로 확신하면 그 방향을 제외한 나머지 방향은 모두 $p_y$값이 작아지고 NLL이 급상승)

⇒ label smoothing이 confidence의 평균은 낮출 수 있지만, 분산을 크게 줄여서 test NLL을 낮춰주고 ‘일반화 성능이 향상’된다고 말하는 것! 

⇒ 예를들어, “label smoothing 적용 후 NLL이 3만큼 감소 → 일반화 성능 향상” 

---

## 3) ECE(Expected Calibration Error) = “확률이 현실을 얼마나 잘 반영하나”

- 이상적인 모델이라면…
    - “이 sample이 정답일 확률이 0.8이다”라고 말하는 sample들을 모아보면
    - 그 중 정답의 비율이 80%정도가 나와야한다.

- 현실에선…
    - “0.9 확신”이라고 했지만 실제론 70%만 맞추는 모델 → “over-confident”
    - “0.6 확신”이라고 했지만 실제론 80% 맞추는 모델 → “under-confident”

---

### 계산 방법)

1. 예측한 max softmax값(=confidence)을 기준으로 [0, 1]구간을 여러 bin으로 나눔
2. 각 bin마다 그 bin에 속한 샘플들의 평균 confidence와 실제로 정답을 맞춘 acc의 차이를 계산
3. 이 값을 전체 sample 비율로 가중 평균해서 구한다. 

$$
ECE = \sum\limits_{b=1}^B \frac {n_b}{N} \times \left\vert acc_b-conf_b \right\vert
$$

(n_b: bin b에 있는 샘플 수, acc_b: 그 bin의 실제 정확도, conf_b: 그 빈의 평균 confidence)

---

- ECE = 0이면 완벽하게 calibrated
- ECE가 크면 모델이 자신의 확률과 현실을 동기화하지 못한 상태
- label smoothing의 큰 장점으로 알려진 “성능 유지 + ECE 감소” 효과를 확인할 수 있다.

## 4) Feature Intra-class Distance “같은 class안에서 feature들이 서로 얼마나 가까운가?”

- 계산방법
    1. class별로 평균 벡터 계산
    2. 같은 class의 각 feature까지의 거리 평균 계산
    3. 이걸 모든 class에 대해 평균 계산
- intra가 작을수록 = 같은 class끼리 tight하게 모여있다.
- intra가 클수록 = 같은 class끼리 variance가 크다.
- Label Smoothing은 internel representation을 더 조밀하고 규칙적으로 만들기 때문에 intra-class distance가 줄어드는 경향이 있음을 확인.

---

그냥 분산과의 차이점

| 항목 | 분산(variance) | Intra-class distance |
| --- | --- | --- |
| 거리 계산 | 제곱 거리 | 유클리드 거리 |
| 값의 단위 | 거리² | 거리 |
| 해석 | 평균에서 얼마나 퍼져 있는가 | 평균에서 얼마나 멀리 떨어져 있는가 |

→ DL의 feature space는 보통 차원이 크기 때문에 분산이 너무 커져서 비교가 어렵거나 차원 수에 따라 값이 민감하게 커진다. 이에 직관적인 “평균 거리”를 사용한다. 

---

## 5) Feature Inter-class Distance “서로 다른 class 간의 중심(centroid) 거리가 얼마나 떨어져 있는가?”

- inter가 크면 = class끼리 멀리 떨어져 있다.
- inter가 작으면 = class끼리 겹치는 부분이 많다.
- 추가로 자주 쓰는 지표:
    
    $$
    ratio = \frac {intra}{inter}
    $$
    
    → 낮을수록 데이터셋의 seperability가 좋은 것. 
    

## 6) Entropy “예측 분포의 불확실성”

- SoftMax 확률 분포의 불확실성:
    
    $$
    Entropy(i)=-\sum\limits_{k}p_{i,k}logp_{i, k}
    $$
    
- entropy가 작으면 = 확실함 → Confidence 높음
- entropy가 크면 = 애매함 → Confidence 낮음
- dataset의 Noise가 많으면 entropy 상승, separability가 높으면 entropy 하락

## 7) Margin “(top1 - top2) 확률의 차”

- 모델이 예측한 가장 높은 확률과 두 번째 확률의 차이
- margin이 클수록 = Confidence가 높다.
- margin이 작을수록 = Confidence가 작다.
- dataset의 separability가 높으면 margin 상승

# 4-1. 간단한 1D ResNet

| **Dataset** | **Method** | **Test Acc** | **Test F1** | **Test NLL** | **Test ECE** | **Max Conf** | **Entropy** | **Margin** | **Intra Dist** | **Inter Dist** | **Sep Ratio** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **UCI** | No LS | 0.9131 | 0.9132 | 0.4131 | 0.0384 | 0.9413 | 0.1244 | 0.8847 | 7.6145 | 24.4290 | 0.3117 |
|  | LS 0.01 | 0.9023 | 0.9004 | 0.4185 | 0.0505 | 0.9223 | 0.2075 | 0.8540 | 6.7245 | 21.8218 | 0.3082 |
| **WISDM** | No LS | 0.9613 | 0.9465 | 0.1228 | 0.0059 | 0.9629 | 0.1080 | 0.9306 | 4.9608 | 21.6525 | 0.2291 |
|  | LS 0.01 | 0.9830 | 0.9711 | 0.0657 | 0.0123 | 0.9744 | 0.1082 | 0.9580 | 3.6752 | 17.9803 | 0.2044 |
| **PAMAP2** | No LS | 0.9794 | 0.9744 | 0.0695 | 0.0104 | 0.9748 | 0.0822 | 0.9575 | 7.0178 | 23.1262 | 0.3035 |
|  | LS 0.01 | 0.9666 | 0.9619 | 0.1033 | 0.0218 | 0.9519 | 0.1823 | 0.9244 | 6.0446 | 18.6980 | 0.3233 |
| **MHEALTH** | No LS | 0.9907 | 0.9917 | 0.0427 | 0.0052 | 0.9917 | 0.0408 | 0.9847 | 3.9070 | 27.2827 | 0.1432 |
|  | LS 0.01 | 0.9907 | 0.9917 | 0.0547 | 0.0132 | 0.9786 | 0.1156 | 0.9660 | 3.5750 | 22.9015 | 0.1561 |

### 1) 결과 요약

- **UCI / PAMAP2 / MHEALTH**: LS(0.01) → **성능/칼리브레이션 대체로 악화**
- **WISDM**: LS(0.01) → **성능/안정성 모두 눈에 띄게 개선**

**“LS는 HAR에서 무조건 좋은 게 아니라, 데이터셋 특성에 따라 이득/손해가 갈린다”**

### 2) 데이터셋별 결과 분석

1. UCI-HAR
- acc: **0.9131 → 0.9023 (-1.1%)**
- f1: **0.9132 → 0.9004 (-1.3%)**
- 이론적으로 기대하던 “LS = 너무 확신하는 거 눌러줌” 패턴은 잘 보임

→ UCI는 원래도 아주 noisy하지 않고, 1D-ResNet이 이미 꽤 잘 학습한 상태라

**추가적인 LS(0.01)는 오히려 일반화/칼리브레이션 둘 다 해침.**

1. WISDM
- acc: **0.9613 → 0.9830 (+2.17%)**
- f1: **0.9465 → 0.9711 (+2.46%)**
- 잘못된 over-confidence를 줄인다기보다는
- **정답에 더욱 confident한 모델로 바꿔줬다**는 느낌.

→ LS가 들어가면서 **같은 클래스끼리는 더 뭉치고, 전체 space는 살짝 압축된 모습**

→ 결과적으로 separability ratio 측면에선 “더 잘 분리되는 공간” 형성.

→ WISDM처럼 **센서 노이즈/subject variance 큰 데이터셋**에서 LS는**overfit된 boundary를 완화하고, 더 안정적인 feature/decision을 만들어줘서 F1, NLL, intra/inter 측면에서 모두 이득**을 줌.

1. PAMAP2
- acc: 0.9794 → 0.9666 (**1.28%**)
- f1: 0.9744 → 0.9619 (**1.25%**)

→ PAMAP2는 이미 매우 잘 분리된 고품질 멀티 센서 데이터라 **LS가 추가적인 이득을 못 주고, 오히려 boundary를 흐리게 만들어 “좋았던 걸 깎아먹은” 케이스.**

1. MHEALTH
- acc: 0.9907 → 0.9907 (**동일**)
- f1: 0.9917 → 0.9917 (**동일**)

→ 정확도는 그대로인데, LS 때문에 **확률 분포만 더 “애매하게” 만들어버린 상황**. 그래서 NLL/ECE 관점에서는 오히려 손해.

→ 이미 거의 완벽한 데이터셋+모델 조합이라 **LS는 “쓸 이유가 없는” 대표 사례**라고 볼 수 있음.

# 4-2. ModernTCN

| **Dataset** | **Method** | **Test Acc** | **Test F1** | **Test NLL** | **Test ECE** | **Max Conf** | **Entropy** | **Margin** | **Intra Dist** | **Inter Dist** | **Sep Ratio** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **UCI** | No LS | 0.9253 | 0.9255 | 0.2304 | 0.0291 | 0.9511 | 0.1245 | 0.9048 | 5.3205 | 15.4016 | 0.3454 |
|  | LS 0.01 | 0.9365 | 0.9372 | 0.2109 | 0.0182 | 0.9396 | 0.1919 | 0.8906 | 5.2009 | 15.0870 | 0.3447 |
| **WISDM** | No LS | 0.9797 | 0.9712 | 0.0834 | 0.0087 | 0.9860 | 0.0413 | 0.9735 | 7.1944 | 14.7445 | 0.4879 |
|  | LS 0.01 | 0.9841 | 0.9746 | 0.0753 | 0.0093 | 0.9776 | 0.1005 | 0.9630 | 5.5618 | 15.4004 | 0.3611 |
| **PAMAP2** | No LS | 0.9743 | 0.9694 | 0.0740 | 0.0095 | 0.9765 | 0.0808 | 0.9606 | 5.8872 | 15.7492 | 0.3738 |
|  | LS 0.01 | 0.9833 | 0.9808 | 0.0756 | 0.0223 | 0.9693 | 0.1407 | 0.9534 | 5.3519 | 15.4567 | 0.3463 |
| **MHEALTH** | No LS | 0.9926 | 0.9932 | 0.0504 | 0.0089 | 0.9837 | 0.0808 | 0.9724 | 4.4354 | 15.9095 | 0.2788 |
|  | LS 0.01 | 0.9926 | 0.9932 | 0.0461 | 0.0165 | 0.9787 | 0.1231 | 0.9679 | 4.4494 | 15.6718 | 0.2839 |
- UCI/WISDM/PAMAP2: F1, ACC가 상승, NLL은 항상 조금 감
- **UCI / WISDM / PAMAP2**:
    
    → **Acc/F1 실제로 올라감** (+0.3~1.1%p 정도)
    
    → NLL은 거의 항상 **조금 감소**(= 정답쪽 확률 배분이 더 “건강해진” 느낌)
    
    → mean max confidence / margin은 **조금 내려가고**, entropy는 **올라감**
    
    → intra-class distance ↓ (클러스터 더 촘촘) → 특히 WISDM, PAMAP2는 꽤 크게 감소
    
    → inter-class는 대부분 비슷하거나 약간만 변동, separability ratio는 대체로 ↓(좋은 방향)
    
- **MHEALTH**:
    - Acc/F1 **완전 동일**
    - NLL만 소폭 개선, ECE는 오히려 악화
    - feature geometry(거리)는 거의 안 변함 / 약간 나빠진 정도

→ **ModernTCN에서는 LS=0.01이 “대부분의 데이터셋에서 성능에 도움” + “confidence는 적당히 낮춤” 패턴**,

MHEALTH는 이미 너무 잘 맞춰서 더 좋아질 구석이 거의 없는 상황처럼 보임.

1. UCI-HAR
- Acc: **0.9253 → 0.9365** (+1.12%p)
- F1: **0.9255 → 0.9372** (+1.17%p)

→ ModernTCN이 UCI에서는 꽤 복잡한 패턴을 잘 학습하고 있고, LS가 과도한 over-confidence를 약간 낮추면서 decision boundary 자체는 더 잘 generalize 되도록 밀어준 상태

1. WISDM
- Acc: **0.9797 → 0.9841** (+0.44%p)
- F1: **0.9712 → 0.9746** (+0.34%p)

→ LS 효과가 제일 “예쁘게” 나온 케이스 중 하나. 

→ classification 성능도 올라가고

→ NLL도 내려가고

→ feature space에서는 **intra ↓, inter ↑ → 클래스 간 분리가 더 명확**

1. PAMAP2
- Acc: **0.9743 → 0.9833** (+0.90%p)
- F1: **0.9694 → 0.9808** (+1.14%p) → 꽤 큰 상승

→ LS가

**→ 성능(F1/Acc)을 꽤 많이 올려주고**

→ feature-level에서도 intra를 확 줄여서 **클래스 내부 cohesion**을 높여줬는데,

→ calibration 측면(ECE)은 오히려 좀 깨진 케이스.

1. MHEALTH
- Acc: **0.9926 → 0.9926** (완전 동일)
- F1: **0.9932 → 0.9932** (완전 동일)

→ 이미 거의 “포화 상태”라 LS를 넣어도 **성능 면에서는 더 좋아질 데가 없는 상황**에 가까움.

# 4-3. 1DResNet vs. ModernTCN 비교

**ModernTCN의 capacity가 더 크고, temporal modeling이 더 잘돼서**

- 복잡하거나 노이즈가 좀 있는 UCI/WISDM/PAMAP2에서
- LS가 “과학습 + 과신(over-confidence)”을 눌러주는 데 잘 작동

MHEALTH의 경우 이미 성능이 포화 상태에 가까워, LS를 적용해도 성능 변화는 없었다.
