from ultralytics import YOLO

import cv2 as cv
import numpy as np
import math

# YOLOv8 모델 로드
model = YOLO('model/aiclops_best.pt')

QR_SIZE = 0.06  # m

# QR 코드 탐지 및 중심 좌표 계산 함수
def detect_with_yolo(image, boxes, camera_matrix, dist_coeffs):

    data = "None"
    qr_center = None
    frame_center = None
    distance_z = None

    for box in boxes:
        xyxy = box.xyxy.cpu().detach().numpy().tolist()[0]
        confidence = box.conf.cpu().detach().numpy().tolist()
        class_id_list = box.cls.cpu().detach().numpy().tolist()

        # class_id_list가 비어있지 않다면 첫 번째 요소 사용
        if class_id_list:
            class_id = int(class_id_list[0])
        else:
            class_id = None  # 기본값 설정

        # b-box 좌표 추출
        x1, y1, x2, y2 = map(int, xyxy)

        # YOLO b-box 가로, 세로 pixel 크기 구하기
        b_width = abs(x2 - x1)
        b_height = abs(y2 - y1)

        # YOLO b-box이자 QR의 중심 좌표 계산
        qr_center = ((x1 + x2) // 2, (y1 + y2) // 2)

        # Camera 화면 center 좌표 계산
        frame_center = (image.shape[1] // 2, image.shape[0] // 2)
        
        # Camera center 표시
        cv.circle(image, frame_center, 5, (255, 255, 255), -1)
        cv.putText(image, "Camera Center", (frame_center[0] + 10, frame_center[1]),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        
        # QR 코드 중심과 카메라 중심 거리 계산 - (x, y) 형식 (단위 : Pixel)
        center_distance_x = qr_center[0] - frame_center[0]
        center_distance_y = qr_center[1] - frame_center[1]

        print(f"Circle center <-> Camera center Distance : {center_distance_x}, {center_distance_y} px")


        #! camera와 QR의 distance (z값)
        # QRcode 크기에 따라 distance 실시간 출력
        
        # 3D 거리 추정: b-box 크기와 QR 코드 실제 크기 사용
        focal_length = camera_matrix[0, 0]  # Focal length from camera matrix
        qr_pixel_size = (b_width + b_height) / 2  # 사각형 평균 크기 (pixel)
        if qr_pixel_size > 0:
            distance_z = (QR_SIZE * focal_length) / qr_pixel_size
        else:
            distance_z = None


        # 결과 출력
        cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2) # YOLO b-box

        cv.circle(image, qr_center, 5, (0, 255, 255), -1)   # QR center point

        cv.putText(image, f"Circle Center : {qr_center}", (qr_center[0] + 10, qr_center[1]),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        cv.putText(image, f"Distance to center (x, y) : ({center_distance_x}, {center_distance_y})", (10, 80), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        
        cv.putText(image, f"B-Box Width: {b_width}, B-Box Height: {b_height}", (10, 110),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        
        cv.putText(image, f"Distance (z): {distance_z:.2f}m", (qr_center[0], qr_center[1] + 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return image, qr_center, data, distance_z

# 실시간 카메라 QR 코드 탐지
def camera_qr_detection():
    cap = cv.VideoCapture(0)  # 내장 카메라 사용
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return
    
    #! camera matrix help~
    # 임시 camera matrix & 왜곡 계수 (삼각법 활용 위해서는 실제 camera calibration 필요)
    camera_matrix = np.array([[1000, 0, 640],
                              [0, 1000, 360],
                              [0, 0, 1]], dtype=float)
    dist_coeffs = np.zeros((4, 1))  # 왜곡 계수 초기화

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break
        
        # YOLOv8 모델로 객체 탐지
        results = model.predict(frame)


        annotated_frame = results[0].plot()
        # b-box 정보 알고 싶으면, 아래와 같이.
        boxes = results[0].boxes

        # 바운딩 박스를 기반으로 QR 코드 중심 좌표 계산
        annotated_frame, qr_center, data, distance_z = detect_with_yolo(annotated_frame, boxes, camera_matrix, dist_coeffs)
        #annotated_frame = detect_qr_with_yolo(annotated_frame, boxes)

        # 결과 출력
        cv.imshow('AICLOPS_Hyun : YOLOv8 Detection', annotated_frame)

        # 'q' 키를 누르면 종료
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    camera_qr_detection()