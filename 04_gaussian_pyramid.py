import numpy as np # 추가
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--image_name', type=str, default='sonoma')
parser.add_argument("--target_size", type=int, nargs=2, default=(240, 135))
parser.add_argument('--kernel_size', type=int, default=5)
parser.add_argument('--sigma', type=float, default=1.0)
args = parser.parse_args()

# --- 01_smoothing.py에서 가져온 함수들 ---
def create_gaussian_kernel(kernel_size=5, sigma=1.0):
    center = kernel_size // 2
    x, y = np.mgrid[-center:center+1, -center:center+1]
    exponent = -(x**2 + y**2) / (2 * sigma**2)
    kernel = np.exp(exponent) / (2 * np.pi * sigma**2)
    kernel = kernel / np.sum(kernel)
    return kernel

def padding(source, pad_size):
    h, w, c = source.shape
    target = np.zeros((h + 2 * pad_size, w + 2 * pad_size, c), dtype=source.dtype)
    target[pad_size : pad_size + h, pad_size : pad_size + w, :] = source
    return target

def filtering(source, kernel):
    h, w, _ = source.shape
    kernel_size = kernel.shape[0]
    pad_size = kernel_size // 2
    source_ = padding(source, pad_size)
    target = np.zeros_like(source, dtype=np.float64)
    for i in range(h):
        for j in range(w):
            for c in range(source.shape[2]):
                patch = source_[i : i + kernel_size, j : j + kernel_size, c]
                target[i, j, c] = np.sum(patch * kernel)
    target = np.clip(target, 0, 255).astype(np.uint8)
    return target
# --- 여기까지 ---

def create_gaussian_pyramid(source, target_size, kernel_size, sigma):
    # -------------------------------------------------------------------------
    # Create Gaussian Pyramid
    #     For each level of pyramid
    #     -- smooth source image
    #     -- downsample source by half 
    # Args:
    #     source (numpy.ndarray): (old_h, old_w, 3)
    #     target_size (tuple): (2,) 
    #     -- target_size[0]: new_h 
    #     -- target_size[1]: new_w
    # Return:
    #     targets (list): [(numpy.ndarray), ..., (numpy.ndarray)]
    #     -- from target image at fine scale to target image coarse scale 
    # -------------------------------------------------------------------------

    targets = [source]
    
    kernel = create_gaussian_kernel(kernel_size, sigma)

    current_image = source

    while current_image.shape[0] > target_size[0] and current_image.shape[1] > target_size[1]:

        smoothed_image = filtering(current_image, kernel)

        downsampled_image = smoothed_image[::2, ::2]

        targets.append(downsampled_image)
        current_image = downsampled_image

    return targets


def main(args):
    # Load image
    source_path = f'{args.image_name}.png'
    source = cv2.imread(source_path)

    # downsampling
    targets = create_gaussian_pyramid(source, args.target_size, args.kernel_size, args.sigma) # 인자 전달 수정
    for level, target in enumerate(targets): # 인덱스와 값의 순서가 바뀜
        target_path = f'{args.image_name}_gaussian_pyramid_level_{level}.png'
        cv2.imwrite(target_path, target)


if __name__ == "__main__":
    main(args)

