import numpy as np # 추가
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='data')
parser.add_argument('--image_name', type=str, default='sonoma')
parser.add_argument('--kernel_size', type=int, default=7)
parser.add_argument('--sigma', type=float, default=4.0)
args = parser.parse_args()


def create_gaussian_kernel(kernel_size=(3, 3), sigma=1.0):
    # -------------------------------------------------------------------------
    # Create Gaussian filter
    # Args:
    #     kernel_size (tuple): (2,) 
    #     -- kernel_size[0]: filter height 
    #     -- kernel_size[1]: filter width
    # Return:
    #     kernel (numpy.ndarray): (new_h, new_w, 3)
    # -------------------------------------------------------------------------

    # Fill this
    # 1. 커널의 중심점을 기준으로 x, y 좌표 그리드 생성
    center = kernel_size // 2
    # np.mgrid: 다차원 그리드 좌표를 생성하는 매우 효율적인 방법
    x, y = np.mgrid[-center:center+1, -center:center+1]

    # 2. 2D 가우시안 함수 적용
    # G(x, y) = (1 / (2 * pi * sigma^2)) * exp(-(x^2 + y^2) / (2 * sigma^2))
    exponent = -(x**2 + y**2) / (2 * sigma**2)
    kernel = np.exp(exponent) / (2 * np.pi * sigma**2)

    # 3. 커널의 모든 요소 합이 1이 되도록 정규화(Normalization)
    # 이렇게 해야 이미지 전체의 밝기가 변하지 않음
    kernel = kernel / np.sum(kernel)
    
    return kernel


def padding(source, pad_size):
    # -------------------------------------------------------------------------
    # Pad zero to boundary 
    # Args:
    #     source (numpy.ndarray): (H, W, 3)
    #     pad_size (int) 
    # Return:
    #     target (numpy.ndarray): (H + 2 * pad_size, W + 2 * pad_size, 3)
    # -------------------------------------------------------------------------

    # 1. 원본 이미지의 높이, 너비, 채널 정보를 가져옴
    h, w, c = source.shape
    
    # 2. 패딩이 추가될 크기의 0으로 채워진 새 배열(틀)을 만듦
    #    (H + 2 * pad_size, W + 2 * pad_size, C)
    target = np.zeros((h + 2 * pad_size, w + 2 * pad_size, c), dtype=source.dtype)
    
    # 3. 새 배열의 중앙에 원본 이미지를 복사해 넣음
    target[pad_size : pad_size + h, pad_size : pad_size + w, :] = source

    return target


def filtering(source, kernel):
    # -------------------------------------------------------------------------
    # Perform linear filtering 
    # Args:
    #     source (numpy.ndarray): (H, W, 3)
    #     filter (numpy.ndarray): (kH, kW) 
    # Return:
    #     target (numpy.ndarray): (H, W, 3)
    # -------------------------------------------------------------------------

    # Create zero padded images
    kernel_size = kernel.shape[0]
    pad_size = kernel_size // 2
    source_ = padding(source, pad_size)

    # 결과 이미지를 저장할 비어있는 행렬 생성
    target = np.zeros_like(source, dtype=np.float64)

    # 컨볼루션(Convolution) 연산 수행
    # 모든 픽셀 위치(i, j)와 모든 채널(c)에 대해 반복
    h, w, _ = source.shape
    for i in range(h):
        for j in range(w):
            for c in range(source.shape[2]):
                # 패딩된 이미지에서 커널 크기만큼의 영역(patch)을 잘라냄
                patch = source_[i : i + kernel_size, j : j + kernel_size, c]

                # patch와 kernel을 요소별로 곱한 뒤, 그 합을 결과 픽셀값으로 저장
                target[i, j, c] = np.sum(patch * kernel)

    # 이미지 저장을 위해 타입을 uint8로 변환
    # 0보다 작거나 255보다 큰 값들을 잘라냄(clipping)
    target = np.clip(target, 0, 255).astype(np.uint8)

    return target


def main(args):
    # Load image
    source_path = f'{args.image_name}.png'
    source = cv2.imread(source_path)

    # Create Gaussian filter
    kernel = create_gaussian_kernel(args.kernel_size, args.sigma)

    # Perform filtering
    target = filtering(source, kernel)

    # Save image
    target_path = source_path = f'{args.image_name}_smooth_k{args.kernel_size}_s{int(args.sigma)}.png'
    cv2.imwrite(target_path, target)


if __name__ == "__main__":
    main(args)
