import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

# 이미지 확인하기
img = cv2.imread('01.jpg')

# plt.figure(figsize=(16, 10))
# plt.imshow(img[:, :, ::-1])
# plt.show()

# 워터마크로 사용할 이미지
img_wm = cv2.imread('watermark_brad2.png')

# plt.imshow(img_wm[:, :, ::-1])
# plt.show()

# 워터마크의 크기를 구하기 
height, width, _ = img.shape
wm_height, wm_width, _ = img_wm.shape

# 워터 마크의 크기는 이미지 보다 작아야됨!!
print(height, width)
print(wm_height, wm_width)

# Encode

# Fast Fourier Transform 주파수 영역으로 변환 
# 변환하면 밝기 값만 나오게되는 데 주파수 영역에 워터마크를 심는다.
img_f = np.fft.fft2(img)

# 허수 영역
print(img_f[0, 0])

# 보안을 위해 암호화 작업
# 워터 마크를 랜덤으로 한 픽셀씩 흩뿌려서 입힌다. 
y_random_indices, x_random_indices = list(range(height)), list(range(width))
# 2021 값은 기억하고 있어야함. 
random.seed(2021)
random.shuffle(x_random_indices)
random.shuffle(y_random_indices)

# 0으로 초기화를 해주고 unsigned integer 8bit테이터 형태로 만들어주기 
random_wm = np.zeros(img.shape, dtype=np.uint8)

# 픽셀들을 하나씩 훑어가면서 랜덤한 위치에다가 워크마크를 흩어트린다. 
for y in range(wm_height):
    for x in range(wm_width):
        random_wm[y_random_indices[y], x_random_indices[x]] = img_wm[y, x]

plt.figure(figsize=(16, 10))
plt.imshow(random_wm)
# input이미지와 같은 크기
plt.show()

# input 이미지에 더해주기 
alpha = 5
# 랜덤한 워터마크에 알파값(5)
result_f = img_f + alpha * random_wm
# iff = invert fast fourier transform 으로 실수 영역으로 돌린다음에 
result = np.fft.ifft2(result_f)
# real : 실수로 변경해주는 코드 
result = np.real(result)
# 타입을 이미지 형태로 변경 
result = result.astype(np.uint8)

fig, axes = plt.subplots(1, 2, figsize=(20, 16))
# 원래 이미지 
axes[0].imshow(img[:, :, ::-1])
axes[0].set_title('Original')
axes[0].axis('off')
# forensic 이미지 
axes[1].imshow(result[:, :, ::-1])
axes[1].set_title('Forensic watermarked')
axes[1].axis('off')
fig.tight_layout()
plt.show()

plt.figure(figsize=(16, 10))
# 사람 눈으로 구분이 안가기 때문에, result를 이미지에 빼기를 할 때 차이가 있음 
plt.imshow(result - img)

# Decode 

# img_ori : 서버에 있는 원본이미지는 워터마크가 심어져 있지 않은 이미지 
# img_input : 불법 녹화 이미지는 워터마크가 심어져 있는 이미지 

# fft를 사용하여 주파수 영역으로 변경해준다. 
img_ori_f = np.fft.fft2(img)
img_input_f = np.fft.fft2(result)

# 원본 - 워터마크 포함된 이미지 = 워터마크 
watermark = (img_ori_f - img_input_f) / alpha
# 실수로 변경하고 다시 이미지화 
watermark = np.real(watermark).astype(np.uint8)

plt.figure(figsize=(16, 10))
plt.imshow(watermark)

y_random_indices, x_random_indices = list(range(height)), list(range(width))
# 위에서 encode할 때와 똑같음. 
random.seed(2021)
random.shuffle(x_random_indices)
random.shuffle(y_random_indices)

result2 = np.zeros(watermark.shape, dtype=np.uint8)

for y in range(height):
    for x in range(width):
        result2[y, x] = watermark[y_random_indices[y], x_random_indices[x]]

# 결과값에서 워터마크가 나옴, 불법 복제되었다는 의미 
plt.figure(figsize=(16, 10))
plt.imshow(result2)

# Result
fig, axes = plt.subplots(1, 3, figsize=(20, 16))
axes[0].imshow(img[:, :, ::-1])
axes[0].set_title('Original')
axes[0].axis('off')
axes[1].imshow(result[:, :, ::-1])
axes[1].set_title('Forensic watermarked')
axes[1].axis('off')
axes[2].imshow(result2)
axes[2].set_title('Watermark')
axes[2].axis('off')
fig.tight_layout()
plt.show()