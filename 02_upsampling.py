import numpy as np # 추가
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--image_name', type=str, default='cat') # 오타 수정
parser.add_argument("--target_size", type=int, nargs=2, default=(2048, 2048))
args = parser.parse_args()


def upsample_nearest_neighbor(source, target_size):
    # -------------------------------------------------------------------------
    # Implement the upsaming algorithm with nearest neighbor interpolation
    # Args:
    #     soruce (numpy.ndarray): (old_h, old_w, 3)
    #     target_size (tuple): (2,) 
    #     -- target_size[0]: new_h 
    #     -- target_size[1]: new_w
    # Return:
    #     target (numpy.ndarray): (new_h, new_w, 3)
    # -------------------------------------------------------------------------

    old_h, old_w, c = source.shape
    new_h, new_w = target_size

    # 새 이미지의 각 픽셀에 대응하는 원본 이미지의 좌표 계산
    y_coords, x_coords = np.mgrid[0:new_h, 0:new_w]

    # 좌표 스케일링
    scale_h = old_h / new_h
    scale_w = old_w / new_w
    old_y_coords = y_coords * scale_h
    old_x_coords = x_coords * scale_w

    # 가장 가까운 원본 픽셀을 찾기 위해 좌표 반올림
    nearest_y = np.round(old_y_coords).astype(np.int32)
    nearest_x = np.round(old_x_coords).astype(np.int32)
    
    # 계산된 좌표가 원본 이미지 크기를 벗어나지 않도록 조정
    nearest_y = np.clip(nearest_y, 0, old_h - 1)
    nearest_x = np.clip(nearest_x, 0, old_w - 1)

    # 계산된 좌표 배열을 이용해 원본 이미지에서 픽셀 값을 한 번에 가져옴
    target = source[nearest_y, nearest_x]
    
    return target


def upsample_bilinear(source, target_size):
    # -------------------------------------------------------------------------
    # Implement the upsaming algorithm with bilienar interpolation
    # Args:
    #     source (numpy array): (old_h, old_w, 3)
    #     target_size (tuple): (2,) 
    #     -- target_size[0]: new_h 
    #     -- target_size[1]: new_w
    # Return:
    #     target (numpy.ndarray): (new_h, new_w, 3)
    # -------------------------------------------------------------------------

    old_h, old_w, c = source.shape
    new_h, new_w = target_size
    
    # 계산의 정확도를 위해 float 타입으로 빈 결과 이미지 생성
    target = np.zeros((new_h, new_w, c), dtype=np.float64)

    scale_h = old_h / new_h
    scale_w = old_w / new_w

    # 새 이미지의 모든 픽셀(y, x)에 대해 반복
    for y in range(new_h):
        for x in range(new_w):
            # 1. 원본 이미지에 대응하는 float 좌표 계산
            y_f = y * scale_h
            x_f = x * scale_w

            # 2. 보간에 사용할 4개의 주변 픽셀 좌표 계산
            y1 = int(np.floor(y_f))
            x1 = int(np.floor(x_f))
            y2 = min(y1 + 1, old_h - 1) # 경계값 처리
            x2 = min(x1 + 1, old_w - 1) # 경계값 처리

            # 3. 4개 픽셀 사이의 거리 비율(가중치) 계산
            dy = y_f - y1
            dx = x_f - x1

            # 4. 양선형 보간법 (Bilinear Interpolation) 수행
            # 4-1. 윗 줄 두 픽셀을 x축 방향으로 선형 보간
            top_val = (1 - dx) * source[y1, x1] + dx * source[y1, x2]
            # 4-2. 아랫 줄 두 픽셀을 x축 방향으로 선형 보간
            bottom_val = (1 - dx) * source[y2, x1] + dx * source[y2, x2]
            # 4-3. 위아래 두 보간 값을 y축 방향으로 선형 보간
            final_val = (1 - dy) * top_val + dy * bottom_val
            
            target[y, x] = final_val

    # 이미지 저장을 위해 타입을 uint8로 변환
    return np.clip(target, 0, 255).astype(np.uint8)


def main(args):
    # Load image
    source_path = f'{args.image_name}.png'
    source = cv2.imread(source_path)

    # x8 upsampling with nearest neighbor interpolation
    target_nn = upsample_nearest_neighbor(source, args.target_size)
    target_nn_path = f'{args.image_name}_upsample_nn.png'
    cv2.imwrite(target_nn_path, target_nn)

    # x8 upsampling with bilinear interpolation
    target_bilinear = upsample_bilinear(source, args.target_size)
    target_bilinear_path = f'{args.image_name}_upsample_bilinear.png'
    cv2.imwrite(target_bilinear_path, target_bilinear)

if __name__ == "__main__":
    main(args)
