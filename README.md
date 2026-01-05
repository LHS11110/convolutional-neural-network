# 합성곱 신경망 구현
본 리포지토리에서는 합성곱 신경망의 원리를 **수식**으로 파악하고, 이를 Python 언어로 구현한다. 만약 합성곱 신경망에 대한 시각적 이해를 위해 찾아온 것이라면 다른 글을 보는 것을 추천한다.
## 합성곱 신경망의 구조
합성곱 신경망을 알려면 먼저 input이 (높이, 너비, 채널)로 주어짐을 알아야 한다. 이는 각각을 하나의 차원으로 생각해볼 수 있으며 따라서 입력은 3차원 텐서가 된다. 높이와 너비는 픽셀 수를 의미하고 채널은 색 성분을 의미한다. 예를 들어 RGB로 색감을 표현하는 $128 \times 128$ 크기의 이미지는 높이 128, 너비 128, (빨강, 초록, 파랑)으로 색감을 표현하므로 채널은 3이 된다.
### 합성곱 연산
합성곱 연산에는 input과 kernel 두 개념이 필요하다. kernel은 filter라고도 불리며 input이 $n \times m \times l$이라면 $x \leq n, y \leq m$에 대해 kernel은 크기가 $x \times y$이고, 개수는 총 $l$개인 행렬들의 모임이다. 따라서 kernel은 $x \times y \times l$인 3차원 텐서라고도 할 수 있겠다. kernel의 각 행렬은 채널 차원을 가지지 않으며 각 높이와 너비 차원의 길이가 input의 높이, 너비 차원의 길이보다 짧아야 하고, 그 크기가 모두 같아야 한다. 여기서 커널의 높이와 너비가 대부분 홀수인 이유는 중앙을 만들어주기 위해서다.
kernel은 보통 $3 \times 3$ 또는 $5 \times 5$의 크기를 갖도록 구현한다. 합성곱 연산을 거친 후의 결과물을 특성 맵(feature map)이라고 하며, 이 또한 마찬가지로 채널 차원을 갖지 않고 높이와 너비 차원만 주어진다. <br>
$$ O_{i, j, k} = \sum^{C - 1}_{l = 0} \sum^{K_w - 1}_{m = 0} \sum^{K_h - 1}_{n = 0} I_{S \cdot i + n, S \cdot j + m, l} \cdot K_{n, m, l, k} $$
여기서 $O, I, K$는 각각 feature map, input, kernel을 의미하고, $C$는 채널 수, $K_h$와 $K_w$는 각각 kernel의 높이와 너비를 의미한다. 이 식에서 볼 수 있듯 3차원 텐서인 kernel은 여러 개가 주어질 수 있으며, kernel의 개수가 1인 경우 특성맵은 그 결과가 2차원 행렬로 나타나게 된다. 위 식에서 kernel을 의미하는 $K$가 4차원 텐서처럼 나타나는 이유는 그냥 여러 개의 kernel 중 $k$번째 kernel을 나타내기 위함이다. 즉, $K$는 그냥 kernel이 아니라 kernel들의 집합이다. $S$는 스트라이드(stride)를 의미한다.
### 특성 맵(feature map)의 크기
$$ \begin{aligned}
O_w &= I_w - K_w + 1 \\
O_h &= I_h - K_h + 1 \\
O_c &= K_c
\end{aligned} $$
위 식은 특성 맵의 크기가 다른 변수들과 어떤 관계에 있는지 보여준다. $w$는 너비, $h$는 높이, $c$는 채널을 의미한다. $K_c$는 단순히 kernel의 개수를 의미한다고 볼 수 있다. 물론 위 식은 $S = 1$이고 zero-padding을 하지 않은 경우에만 성립한다. 여기서 $S$가 $1$이 아닌 경우까지 포함하는 경우에는 다음과 같다.
$$ \begin{aligned}
O_w &= \mathrm{floor}(\frac{I_w - K_w}{S}) + 1 \\
O_h &= \mathrm{floor}(\frac{I_h - K_h}{S}) + 1 \\
O_c &= K_c
\end{aligned} $$
여기에 zero-padding까지 고려하면 다음과 같다. $P$는 패딩의 폭이다.
$$ \begin{aligned}
O_w &= \mathrm{floor}(\frac{I_w - K_w + 2P}{S}) + 1 \\
O_h &= \mathrm{floor}(\frac{I_h - K_h + 2P}{S}) + 1 \\
O_c &= K_c
\end{aligned} $$
커널 집합 $K$는 신경망 입장에서 학습되어야 하는 가중치들의 집합에 불과하다. 이 가중치들의 개수는 다음과 같다.
$$ K_w \times K_h \times I_c \times K_c $$
### 패딩(padding)과 편향(bias)
아래는 패딩의 폭이 $1$인 zero-padding이 무엇인지 보여준다.
$$ \begin{pmatrix}
a_{00} & a_{01} & \cdots & a_{0m} \\
a_{10} & a_{11} & \cdots & a_{1m} \\
\vdots & \vdots & \ddots & \vdots \\
a_{n0} & a_{n1} & \cdots & a_{nm}
\end{pmatrix} \rightarrow \begin{pmatrix}
0 & 0 & 0 & \cdots & 0 & 0 \\
0 & a_{00} & a_{01} & \cdots & a_{0m} & 0 \\
0 & a_{10} & a_{11} & \cdots & a_{1m} & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & a_{n0} & a_{n1} & \cdots & a_{nm} & 0 \\
0 & 0 & 0 & \cdots & 0 & 0
\end{pmatrix} $$
합성곱 신경망의 편향(bias)은 다음 스칼라 덧셈의 정의를 통해 다루어진다. 스칼라 $a$와 행렬 $A$에 대해
$$ (a + A)_{ij} = a + A_{ij} $$
편향은 kernel이 적용된 후 결과인 feature map 행렬에 연산되어진다. 따라서 편향의 개수는 커널의 개수와 같다.
$$ O_{i, j, k} = \left(\sum^{C - 1}_{l = 0} \sum^{K_w - 1}_{m = 0} \sum^{K_h - 1}_{n = 0} I_{S \cdot i + n, S \cdot j + m, l} \cdot K_{n, m, l, k}\right) + b_k $$
### 풀링(pooling)
합성곱 신경망은 일반적으로 합성곱 층(합성곱 연산 + 활성화 함수) 다음에 풀링 층을 추가하여 구현된다. 풀링 층은 feature map의 크기를 줄이는 역할을 하는 층으로 보통 두 함수 $\mathrm{max}, \mathrm{avg}$를 통해 구현된다. pooling 또한 합성곱 층의 kernel, stride 개념을 사용하며, $I$를 합성곱 층을 통과한 결과인 3차원 텐서, $O$를 pooling 연산 후의 결과인 행렬이라고 했을 때 $\mathrm{max}$를 사용한 정의는 다음과 같다.
$$



$$
풀링 연산은 채널 수를 그대로 둔 채 너비와 높이를 줄이는 연산이라고도 볼 수 있다. 이는 각 feature map은 kernel에 의해 특징이 증폭된 값들의 행렬이므로 kernel의 개수만큼 있어야 하는 것이 그 이유라고 볼 수 있다.