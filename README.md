**1. 프로젝트의 목적**

주제: 손글씨 숫자 및 문자 인식 시스템
배경: 손글씨 숫자와 문자 인식은 머신러닝 분야에서 고전적이고도 중요한 문제입니다. 이는 우편번호 인식, 서류 스캔, 디지털 필기 입력 등 다양한 응용 분야에 활용됩니다.
목적: 딥러닝 기술을 활용하여 MNIST 데이터셋(숫자)과 EMNIST 데이터셋(문자)의 손글씨 데이터를 학습시켜, 주어진 이미지를 정확히 분류하는 모델을 구축하고 이를 확장 가능하도록 설계했습니다.

**2. 데이터셋**

MNIST 데이터셋:
  손글씨 숫자 (0~9)의 이미지를 포함한 데이터셋.
  28x28 픽셀, 흑백 이미지.
  60,000개의 훈련 데이터와 10,000개의 테스트 데이터.
EMNIST 데이터셋:
  손글씨 문자 및 숫자를 포함한 확장 데이터셋.
  다양한 문자 집합 지원 (예: 대문자/소문자/숫자).
데이터 전처리 과정:
  각 이미지는 28x28 크기의 배열로 정규화.
  레이블은 원-핫 인코딩으로 변환하여 분류 문제를 해결.

**3. 구현한 시스템 구조(코드 설명 단계별 상세 내용)**

 (1) 데이터 로드 및 전처리
  TensorFlow와 Keras를 사용하여 데이터셋을 로드.
  데이터는 [0, 1] 범위로 정규화.
  MNIST는 기본으로 제공되지만, EMNIST는 tensorflow-datasets 또는 추가 데이터 로드 방식을 사용.

 (2) 모델 설계
  입력: 28x28 크기의 흑백 이미지.
  CNN (합성곱 신경망) 구조를 사용:
  Conv2D + MaxPooling을 반복하여 이미지 특징 추출.
  Flatten + Dense 층으로 분류기 구성.
  활성화 함수: ReLU와 Softmax 사용.
  손실 함수: SparseCategoricalCrossentropy.

 (3)  모델 학습
  학습 데이터로 훈련, 검증 데이터로 성능 점검.
  Adam 옵티마이저 사용, 학습 속도 향상.

 (4) 결과 평가
  테스트 데이터에서의 정확도 계산.
  혼동 행렬을 통해 모델의 예측 오류 분석.

 (5) 응용 확장
  EMNIST로 문자 인식까지 확장.
  UI를 추가해 실시간 입력(마우스 드로잉) 예측.

**4. 코드의 주요 부분**

 (1) 데이터 로드 및 전처리

# MNIST 데이터셋 로드 및 정규화
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 차원 확장 (채널 추가)
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

 (2) 모델 설계

# CNN 모델 정의
model = models.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')  # MNIST는 10개의 클래스를 분류
])

 (3) 학습 및 평가

# 모델 컴파일 및 학습
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 모델 훈련
model.fit(x_train, y_train, epochs=10, validation_split=0.2)

# 평가
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc}")

**5. 응용 확장 (추가 기능)**

 (1) EMNIST 문자 데이터 추가 학습
  EMNIST 데이터셋으로 문자 분류.
  동일한 CNN 구조 사용 가능.

 (2) 혼동 행렬로 성능 분석

from sklearn.metrics import confusion_matrix
import seaborn as sns

y_pred = model.predict(x_test).argmax(axis=1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

 (3) 실시간 입력 시스템
  Flask 또는 Tkinter로 간단한 UI 구현.
  사용자가 직접 숫자나 문자를 입력하여 모델이 예측.

**6. 결과 및 시사점**

 (1) 결과
  MNIST에서는 약 99% 이상의 높은 정확도를 달성.
  EMNIST에서는 문자 인식으로 확장했을 때도 높은 성능 유지.

 (2) 시사점
  CNN 기반 모델은 이미지 인식 문제에서 탁월한 성능을 발휘함.
  추가적인 데이터셋과 증강을 통해 더 다양한 응용 가능.

 (3) 확장 가능성
  필기 인식 외에도 OCR(문자인식), 얼굴 인식 등에 활용 가능.
  더욱 복잡한 데이터셋(CIFAR-10, CIFAR-100)으로 확장 가능.
