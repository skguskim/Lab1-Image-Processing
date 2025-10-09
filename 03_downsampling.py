import numpy as np #추가
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--image_name', type=str, default='sonoma') # 오타 수정
parser.add_argument("--target_size", type=int, nargs=2, default=(135, 240)) # 사이즈 수정
args = parser.parse_args()

def downsample(source, target_size):
    # -------------------------------------------------------------------------
    # Implement the downsampling algorithm
    #     If the spatial coordinates to be sampled are real numbers, 
    #     perform sampling using nearest neighbor interpolation. 
    # Args:
    #     source (numpy.ndarray): (old_h, old_w, 3)
    #     target_size (tuple): (2,) 
    #     -- target_size[0]: new_h 
    #     -- target_size[1]: new_w
    # Return:
    #     target (numpy.ndarray): (new_h, new_w, 3)
    # -------------------------------------------------------------------------

    old_h, old_w, c = source.shape
    new_h, new_w = target_size

    # 1. 작은 이미지의 각 픽셀에 대응하는 원본 이미지의 좌표를 계산합니다.
    #    (upsample_nearest_neighbor와 완전히 동일한 로직)
    y_coords, x_coords = np.mgrid[0:new_h, 0:new_w]

    # 2. 좌표 스케일링
    scale_h = old_h / new_h
    scale_w = old_w / new_w
    old_y_coords = y_coords * scale_h
    old_x_coords = x_coords * scale_w

    # 3. 가장 가까운 원본 픽셀을 찾기 위해 좌표를 반올림합니다.
    nearest_y = np.round(old_y_coords).astype(np.int32)
    nearest_x = np.round(old_x_coords).astype(np.int32)

    # 4. 좌표가 원본 이미지 크기를 벗어나지 않도록 조정합니다.
    nearest_y = np.clip(nearest_y, 0, old_h - 1)
    nearest_x = np.clip(nearest_x, 0, old_w - 1)

    # 5. 계산된 좌표 배열을 이용해 원본 이미지에서 픽셀 값을 가져와 작은 이미지를 생성합니다.
    target = source[nearest_y, nearest_x]

    return target


def main(args):
    # Load image
    source_path = f'{args.image_name}.png'
    source = cv2.imread(source_path)

    # /8 downsampling
    target = downsample(source, args.target_size)
    target_path = f'{args.image_name}_downsample.png'
    cv2.imwrite(target_path, target)


if __name__ == "__main__":
    main(args)
