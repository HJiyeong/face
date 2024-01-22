import dlib
import cv2
import json
import os
import numpy as np

def extract_landmarks(image_path, predictor):
    image = cv2.imread(image_path)

    detector = dlib.get_frontal_face_detector()
    face = detector(image)[0]



    original_landmarks = predictor(image, face)

    rotated_image = rotate_image(image, original_landmarks)

    landmarks = predictor(rotated_image, face)
    landmarks_info = []

    for i in range(68):
        x, y = landmarks.part(i).x, landmarks.part(i).y
        landmarks_info.append((x, y))

    person_name = os.path.basename(image_path).split('_')[0]
    landmarks_data = []

    landmarks_data.append({
        'image_path': image_path,
        'landmarks': landmarks_info,
        'label': person_name
     })

    # 랜드마크 표시를 메인 함수로 이동
    show_landmarks(rotated_image, landmarks_info)

    return landmarks_data


import cv2
import numpy as np


def rotate_image(image, landmarks):
    # 얼굴 특징점 좌표 추출
    left_eye = (landmarks.part(36).x, landmarks.part(36).y)
    right_eye = (landmarks.part(45).x, landmarks.part(45).y)

    # glabella_center 계산
    gx = (left_eye[0] + right_eye[0]) // 2
    gy = (left_eye[1] + right_eye[1]) // 2
    glabella_center = (gx, gy)

    # 눈썹 영역의 중심을 forehead의 중심으로 사용
    eyebrow_center = ((landmarks.part(17).x + landmarks.part(26).x) // 2, (landmarks.part(17).y + landmarks.part(26).y) // 2)
    forehead_center = (eyebrow_center[0], landmarks.part(19).y)


    # nose 좌표
    nose = (landmarks.part(30).x, landmarks.part(30).y)

    # 삼각형 변 길이 계산
    a = np.sqrt((forehead_center[0] - nose[0]) ** 2 + (forehead_center[1] - nose[1]) ** 2)
    b = np.sqrt((glabella_center[0] - nose[0]) ** 2 + (glabella_center[1] - nose[1]) ** 2)
    c = np.sqrt((glabella_center[0] - forehead_center[0]) ** 2 + (glabella_center[1] - forehead_center[1]) ** 2)

    # 회전각 계산
    angle = np.degrees(np.arccos((a ** 2 + c ** 2 - b ** 2) / (2 * a * c)))


    if forehead_center[0] < nose[0]:  # 얼굴이 오른쪽을 향하고 있을 때
        angle = -angle

    # 이미지 회전
    rotated_image = cv2.warpAffine(image, cv2.getRotationMatrix2D(glabella_center, angle, 1.0),(image.shape[1], image.shape[0]))

    # 회전된 이미지에 점 찍기
    cv2.circle(image, forehead_center, 5, (0, 255, 0), -1)  # 녹색
    cv2.circle(image, glabella_center, 5, (0, 0, 255), -1)  # 빨간색
    cv2.circle(image, nose, 5, (255, 0, 0), -1)  # 파란색

    # 삼각형 그리기
    triangle = np.array([forehead_center, nose, glabella_center], dtype=np.int32)
    cv2.polylines(image, [triangle], isClosed=True, color=(255, 255, 255), thickness=2)

    # 회전된 이미지 출력
    cv2.imshow('Rotated Image with Points and Triangle', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return rotated_image


def show_landmarks(image, landmarks):
    for landmark in landmarks:
        cv2.circle(image, landmark, 3, (0, 255, 0), -1)

    cv2.imshow('Landmarks', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




def save_landmarks_to_json(data, json_file):
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    image_folder = '/Users/hwangjiyeong/PycharmProjects/facial reconition/face_recognition_1/train data'
    output_json_file = '/Users/hwangjiyeong/PycharmProjects/facial reconition/face_recognition_1/landmark/landmark_datas'  # 결과를 저장할 JSON 파일

    # 렌드마크 검출기 초기화
    predictor = dlib.shape_predictor('/Users/hwangjiyeong/PycharmProjects/facial reconition/face_recognition_1/shape_predictor_68_face_landmarks.dat')

    dataset = []

    # 이미지 폴더에서 이미지 파일 목록 읽어오기
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(image_folder, filename)
            landmarks_data = extract_landmarks(image_path, predictor)
            dataset.extend(landmarks_data)

    # 데이터셋을 JSON 파일로 저장
    save_landmarks_to_json(dataset, output_json_file)

