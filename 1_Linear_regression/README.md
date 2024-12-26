# Linear Regression and Multiple Linear Regression Models

이 코드는 간단한 경사 하강법 기반 접근법부터 PyTorch 기반 구현까지 다양한 선형 회귀 모델 구현을 보여줍니다.

---

## 목차
1. [경사 하강법 기반 선형 회귀](#경사-하강법-기반-선형-회귀)
2. [정규 방정식을 사용한 선형 회귀](#정규-방정식을-사용한-선형-회귀)
3. [다항 회귀](#다항-회귀)
4. [PyTorch를 사용한 선형 회귀](#pytorch를-사용한-선형-회귀)
5. [PyTorch를 사용한 다중 선형 회귀](#pytorch를-사용한-다중-선형-회귀)
6. [데이터 세부 정보](#데이터-세부-정보)
7. [결과 요약](#결과-요약)

---

## 경사 하강법 기반 선형 회귀

- **목적**: 간단한 선형 회귀 모델의 최적 가중치(`W`)와 절편(`b`)을 찾는 것.
- **구현**:
  - 가설: `y = W * x + b`
  - 손실 함수: 평균 제곱 오차(MSE)
  - 경사 하강법을 사용하여 비용 최소화.
- **결과**:
  - `y = 2 * x` 데이터셋에서 `W = 2.0`, `b = 0.0`으로 수렴.
  - 비용 감소를 시각화.

---

## 정규 방정식을 사용한 선형 회귀

- **목적**: 정규 방정식을 사용하여 선형 회귀 문제를 해결.
- **구현**:
  - 특징 행렬에 편향 항 추가.
  - `np.linalg.inv`를 사용하여 행렬 역 계산.
  - `theta = (X.T X)^-1 X.T y`로 예측.
- **데이터셋**:
  - 범주형 특징을 제외한 Kaggle 자동차 데이터셋.
- **결과**:
  - 테스트 데이터셋의 평균 제곱 오차(MSE): `4.0242`.

---

## 다항 회귀

- **목적**: 다항 회귀를 사용하여 비선형 데이터를 모델링.
- **구현**:
  - 다항 항목(예: `x^2`)을 포함하도록 특징 행렬 확장.
  - 정규 방정식을 사용하여 해결.
- **데이터셋**:
  - 이차 관계를 가진 합성 데이터셋.
- **결과**:
  - 데이터에 이차 곡선을 적합.

---

## PyTorch를 사용한 선형 회귀

- **목적**: PyTorch를 사용하여 선형 회귀 모델 구현.
- **구현**:
  - `torch.nn.Linear`를 사용하여 선형 모델 생성.
  - SGD 및 MSELoss를 사용하여 최적화.
  - 학습된 회귀선을 시각화.
- **결과**:
  - 학습된 가중치와 절편이 실제 값에 가까워짐.
  - 200 에포크 동안 손실이 꾸준히 감소.

---

## PyTorch를 사용한 다중 선형 회귀

- **목적**: 여러 특징을 사용하여 자동차 판매 가격 예측.
- **구현**:
  - `StandardScaler`를 사용하여 데이터 전처리.
  - 4개의 입력 특징을 가진 PyTorch 모델 정의.
  - MSELoss 및 SGD 옵티마이저를 사용하여 모델 학습.
- **결과**:
  - 학습된 모델이 테스트 손실 `0.1661` 달성.

---

## 데이터 세부 정보

- **출처**: Kaggle에서 제공된 자동차 정보 데이터셋.
- **컬럼 설명**:
  - `Car_Name`: 차량 이름
  - `Year`: 제조 연도
  - `Selling_Price`: 차량 판매 가격
  - `Present_Price`: 현재 시장 가격
  - `Kms_Driven`: 주행 거리 (킬로미터)
  - `Fuel_Type`: 연료 유형 (휘발유/디젤)
  - `Seller_Type`: 판매자 유형 (딜러/개인)
  - `Transmission`: 변속 방식 (수동/자동)
  - `Owner`: 이전 소유자 수
- **전처리**:
  - 범주형 컬럼(`Fuel_Type`, `Seller_Type`, `Transmission`, `Car_Name`) 삭제.
  - `StandardScaler`를 사용하여 숫자형 피처 스케일링.

---

## 결과 요약

- 경사 하강법은 단순 데이터에 효과적으로 수렴.
- 정규 방정식은 작은 데이터셋에 효율적인 솔루션 제공.
- PyTorch 구현은 더 큰 데이터셋과 복잡한 시나리오를 효과적으로 처리.
- 스케일링과 같은 전처리는 모델 성능 최적화에 중요.
