import cv2 as cv
import numpy as np
from sklearn.cluster import KMeans
import glob
import os


#! 1. Augmentation된 보라색 원 이미지 데이터를 학습하여 최적의 threshold 값을 추출
# 보라색 원이 포함된 augmentation된 이미지들의 경로 설정
image_folder = 'output_folder/*.jpg'
image_paths = glob.glob(image_folder)

def process_image(image_path):
    # 이미지 로드 및 흑백 변환
    img = cv.imread(image_path)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 이미지를 1D 배열로 변환
    pixel_values = gray_img.reshape((-1, 1))
    pixel_values = np.float32(pixel_values)

    return pixel_values, img.shape

# 모든 이미지에서 픽셀 데이터를 수집하여 K-means 학습용 데이터셋 생성
def create_training_data(image_paths):
    training_data = []

    for image_path in image_paths:
        pixel_values, _ = process_image(image_path)
        training_data.append(pixel_values)

    # 데이터를 하나로 합침
    training_data = np.vstack(training_data)

    return training_data

training_data = create_training_data(image_paths)

# K-means 클러스터링을 통한 임계값 추출
def kmeans_threshold(training_data, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(training_data)

    # 각 클러스터의 중심값 계산
    centers = kmeans.cluster_centers_

    # 중심값 중 더 높은 값을 threshold로 사용 (보라색 픽셀을 대표)
    threshold = max(centers)
    print(f"추출된 최적의 threshold 값: {threshold[0]}")

    return threshold[0]

threshold = kmeans_threshold(training_data)

#! 2. 추출된 최적의 threshold를 사용하여 테스트 이미지에서 보라색 픽셀 추출
def extract_purple_pixels(image_path, threshold):
    # 이미지 로드 및 흑백 변환
    img = cv.imread(image_path)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # threshold 값을 이용해 보라색 픽셀 추출
    _, mask = cv.threshold(gray_img, threshold, 255, cv.THRESH_BINARY)

    # 원본 이미지에서 보라색 픽셀만 추출
    purple_pixels = cv.bitwise_and(img, img, mask=mask)

    return purple_pixels, mask

# 테스트 이미지에서 보라색 픽셀 추출
test_image_path = 'test_image.jpg'
purple_image, purple_mask = extract_purple_pixels(test_image_path, threshold)

# 추출된 보라색 원 이미지를 저장
cv.imwrite('extracted_purple_image.jpg', purple_image)
cv.imwrite('purple_mask.jpg', purple_mask)

'''
#!3. 허프 변환을 통해 원 탐지
def detect_circle_with_hough(mask_image):
    # 엣지 검출 (Canny Edge Detection)
    edges = cv.Canny(mask_image, 50, 150)

    # 허프 변환을 통한 원 검출
    circles = cv.HoughCircles(edges, cv.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=50, param2=30, minRadius=10, maxRadius=100)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        print(f"검출된 원 개수: {len(circles)}")

        for (x, y, r) in circles:
            # 원을 그려서 표시
            cv.circle(mask_image, (x, y), r, (0, 255, 0), 2)

        return mask_image
    else:
        print("원을 검출하지 못했습니다.")
        return None

# 허프 변환을 통한 원 탐지
detected_circle_image = detect_circle_with_hough(purple_mask)

if detected_circle_image is not None:
    cv.imwrite('detected_circle_image.jpg', detected_circle_image)


#! 4. 캘리브레이션을 통한 원형 보정 및 크기 계산
def calibrate_circle(image, circles):
    # 실제 환경에 맞는 보정을 위한 기본적인 변환 (예시)
    transformation_matrix = np.eye(3)

    for (x, y, r) in circles:
        # 중심과 반지름 보정
        new_center = cv.perspectiveTransform(np.array([[x, y]], dtype="float32"), transformation_matrix)
        new_radius = r  # 반지름 보정이 필요한 경우 추가 가능

        # 보정된 원 표시
        cv.circle(image, (int(new_center[0][0]), int(new_center[0][1])), int(new_radius), (255, 0, 0), 2)

    return image

# 검출된 원을 캘리브레이션
if detected_circle_image is not None:
    calibrated_image = calibrate_circle(purple_image, circles)
    cv.imwrite('calibrated_circle_image.jpg', calibrated_image)
'''