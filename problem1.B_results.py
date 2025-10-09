import cv2
import matplotlib.pyplot as plt

# --- 이미지 파일 이름 설정 ---
IMAGE_NAME = 'sonoma'
SIGMA = 4

# 1. 비교할 이미지들을 불러옴 
try:
    img_k3 = cv2.imread(f'{IMAGE_NAME}_smooth_k3_s{SIGMA}.png')
    img_k7 = cv2.imread(f'{IMAGE_NAME}_smooth_k7_s{SIGMA}.png')
    img_k11 = cv2.imread(f'{IMAGE_NAME}_smooth_k11_s{SIGMA}.png')
    
    # 이미지가 없는 경우를 대비한 에러 처리
    if img_k3 is None or img_k7 is None or img_k11 is None:
        raise FileNotFoundError("이미지 파일을 찾을 수 없음")

except FileNotFoundError as e:
    print(e)
    exit()


# 2. 색상 보정: OpenCV(BGR) -> Matplotlib(RGB)
img_k3_rgb = cv2.cvtColor(img_k3, cv2.COLOR_BGR2RGB)
img_k7_rgb = cv2.cvtColor(img_k7, cv2.COLOR_BGR2RGB)
img_k11_rgb = cv2.cvtColor(img_k11, cv2.COLOR_BGR2RGB)

# 이미지 리스트와 제목 리스트 생성
images = [img_k3_rgb, img_k7_rgb, img_k11_rgb]
titles = ['Kernel Size = 3', 'Kernel Size = 7', 'Kernel Size = 11']

# 3. Plot 생성
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 4. 각 subplot에 이미지와 제목을 추가
for i in range(3):
    axes[i].imshow(images[i])
    axes[i].set_title(titles[i])
    axes[i].axis('off') # 불필요한 x, y축 눈금 제거

# 5. Plot 보여주기
plt.tight_layout() # 이미지 간 간격 자동 조절
plt.show()