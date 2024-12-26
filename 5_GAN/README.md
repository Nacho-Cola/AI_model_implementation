# GAN with MNIST

이 문서는 MNIST 데이터셋을 활용하여 생성적 적대 신경망(GAN)을 구현하고 평가하는 방법을 제공합니다. GAN은 새로운 데이터를 생성하는 데 사용되는 강력한 딥러닝 기술입니다. 이 프로젝트는 숫자 이미지 생성 문제를 해결하는 데 중점을 둡니다.

---

## 목차
1. [개요](#개요)
2. [GAN 구현](#gan-구현)
3. [데이터 세부 정보](#데이터-세부-정보)
4. [결과 요약](#결과-요약)

---

## 개요

이 프로젝트는 TensorFlow/Keras를 사용하여 GAN을 구현합니다. GAN은 Generator(생성자)와 Discriminator(판별자)라는 두 개의 신경망으로 구성됩니다. 이 네트워크는 상호 작용하면서 학습하여 실제와 같은 데이터를 생성합니다.

---

## GAN 구현

### 데이터 전처리

MNIST 데이터셋을 사용하여 GAN을 학습시킵니다. 데이터는 다음과 같이 전처리됩니다:
- 데이터를 (28x28x1) 이미지로 유지.
- 픽셀 값을 [-1, 1] 범위로 정규화.

```python
(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # 픽셀 값을 [-1, 1]로 정규화.

BUFFER_SIZE = 60000
BATCH_SIZE = 256
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
```

### Generator 모델

Generator는 랜덤 노이즈를 입력으로 받아 실제와 유사한 이미지를 생성합니다. 네트워크는 여러 개의 Dense 레이어로 구성되며, 최종 출력은 `(28, 28, 1)` 이미지입니다.

```python
def build_generator():
    model = Sequential([
        Dense(256, activation='relu', input_dim=100),
        BatchNormalization(momentum=0.8),
        Dense(512, activation='relu'),
        BatchNormalization(momentum=0.8),
        Dense(1024, activation='relu'),
        BatchNormalization(momentum=0.8),
        Dense(28*28*1, activation='tanh'),
        Reshape((28, 28, 1))
    ])
    return model

generator = build_generator()
```

### Discriminator 모델

Discriminator는 이미지가 실제 이미지인지, Generator가 생성한 가짜 이미지인지 판별합니다. 여러 Dense 레이어로 구성되며 출력은 단일 값(진짜/가짜)입니다.

```python
def build_discriminator():
    model = Sequential([
        Flatten(input_shape=(28, 28, 1)),
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(1)
    ])
    return model

discriminator = build_discriminator()
```

### 손실 함수

Generator와 Discriminator 모두에 대해 Binary Cross-Entropy 손실을 사용합니다.

```python
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss
```

### 학습 루프

GAN 학습 루프는 다음 단계로 구성됩니다:
1. 랜덤 노이즈를 생성하여 Generator에 입력.
2. Generator가 생성한 이미지를 Discriminator에 입력하여 진짜/가짜 판별.
3. Generator와 Discriminator의 손실을 계산하고, 역전파를 통해 가중치를 업데이트.

```python
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        print(f'Time for epoch {epoch + 1} is {time.time() - start} sec')

train(train_dataset, epochs=300)
```

### 결과 시각화

학습이 진행됨에 따라 Generator가 생성한 이미지를 시각화합니다.

```python
def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig(f'image_at_epoch_{epoch:04d}.png')
    plt.show()
```

---

## 데이터 세부 정보

### MNIST 데이터셋
- **출처**: TensorFlow/Keras의 MNIST 데이터셋.
- **특성**:
  - 28x28 크기의 흑백 이미지.
  - 10개의 숫자 클래스(0부터 9까지).
- **데이터 분할**:
  - 학습 데이터: 60,000개
  - 테스트 데이터: 10,000개

---

## 결과 요약

- GAN은 MNIST 데이터셋을 기반으로 학습되어 실제와 유사한 숫자 이미지를 생성.
- 학습이 진행됨에 따라 생성 이미지의 품질이 점차 향상됨.
- 최종 결과물은 `.gif` 파일로 저장하여 학습 과정 시각화.

