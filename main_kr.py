import cv2
import mediapipe as mp
import time
import numpy as np
import keyboard
from PIL import ImageFont, ImageDraw, Image
from unicode import join_jamos

max_num_hands = 1  # 인식할 손의 갯수

gesture = {  # 손동작
    0: 'ㄱ', 1: 'ㄴ', 2: 'ㄷ', 3: 'ㄹ', 4: 'ㅁ', 5: 'ㅂ', 6: 'ㅅ', 7: 'ㅇ', 8: 'ㅈ', 9: 'ㅊ', 10: 'ㅋ', 11: 'ㅌ', 12: 'ㅍ', 13: 'ㅎ', 14: 'ㅏ',
    15:'ㅑ', 16: 'ㅓ', 17: 'ㅕ', 18: 'ㅗ', 19: 'ㅛ', 20: 'ㅜ', 21: 'ㅠ', 22: 'ㅡ', 23: 'ㅣ', 24: 'ㅐ', 25: 'ㅒ', 26: 'ㅔ', 27: 'ㅖ', 28: "ㅢ",
    29: "ㅚ", 30: "ㅟ", 31: "SPACING", 32: "BACKSPACE", 33: "CLEAR"
}

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,  # 인식할 손의 개수 제한
    min_detection_confidence=0.5,  # 최소 탐지 신뢰값 (기본값 0.5)
    min_tracking_confidence=0.5   # 최소 추적 신뢰값 (기본값 0.5)
)

# 학습 데이터 저장
# f = open('test.txt', 'w')

# 데이터
file = np.genfromtxt('dataSet_kr.txt', delimiter=',')
font_path = "fonts/SUITE-Medium.ttf" # PIL 이미지 폰트
angleFile = file[:, :-1]
labelFile = file[:, -1]
angle = angleFile.astype(np.float32)  # 학습 데이터 행렬
label = labelFile.astype(np.float32)  # 각 학습 데이터에 대응되는 레이블 행렬
knn = cv2.ml.KNearest_create()  # KNN 알고리즘 객체 생성
knn.train(angle, cv2.ml.ROW_SAMPLE, label)  # KNN 알고리즘 학습
cap = cv2.VideoCapture(0)
startTime = time.time()
prev_idx = 0
sentence = ''
recognizeDelay = 1

while True:
    ret, img = cap.read()
    if not ret:  # ret 의 값이 없다면
        continue
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]
            v = v2 - v1  # 벡터 뺄셈 연산
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            compareV1 = v[[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 16, 17], :]
            compareV2 = v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]
            angle = np.arccos(np.einsum('nt,nt->n', compareV1, compareV2))

            angle = np.degrees(angle)
            # 데이터 학습
            # if keyboard.is_pressed('s'):  # s를 누를 시 현재 데이터(angle)가 txt 파일에 저장됨
            #     for num in angle:
            #         num = round(num, 6)
            #         f.write(str(num))
            #         f.write(',')
            #     f.write("32.000000")  # 데이터를 저장할 gesture의 label 번호
            #     f.write("\n")
            #     print('next')
            data = np.array([angle], dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data, 3)
            idx = int(results[0][0])
            if idx in gesture.keys():
                if idx != prev_idx:
                    startTime = time.time()
                    prev_idx = idx
                else:
                    if time.time() - startTime > recognizeDelay:
                        if idx == 31:  # 공백
                            sentence += ' '
                        elif idx == 32:  # 한 글자 삭제
                            sentence = sentence[:-1]
                        elif idx == 33:
                            sentence = " "
                        else:  # 문자=
                            sentence += gesture[idx]
                        startTime = time.time()

                # 화면에 현재 손 모양 문자 표시
                img_pillow = Image.fromarray(img)
                font = ImageFont.truetype(font_path, 24)
                draw = ImageDraw.Draw(img_pillow, 'RGB')

                # 텍스트 입력
                draw.text((int(res.landmark[0].x * img.shape[1] - 10), int(res.landmark[0].y * img.shape[0] + 40)),
                          gesture[idx], font=font, fill=(0, 0, 0))

                # cv2 이미지로 변환
                img = np.array(img_pillow)
                # cv2.putText(img, gesture[idx].upper(), (int(res.landmark[0].x * img.shape[1] - 10),
                #                                         int(res.landmark[0].y * img.shape[0] + 40)), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(0, 0, 0))

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
    # 화면에 현재 문장 표시
    # 한글 사용하기 위해 cv2 이미지를 pillow 이미지로 변환
    img_pillow = Image.fromarray(img)
    font = ImageFont.truetype(font_path, 24)
    draw = ImageDraw.Draw(img_pillow, 'RGB')
    join_sentence = join_jamos(sentence)
    # 텍스트 입력
    draw.text((20, 440), join_sentence, font=font, fill=(0, 0, 0))
    
    # cv2 이미지로 변환
    img_cv2 = np.array(img_pillow)

    cv2.imshow('Sign Language Tracking', img_cv2)
    cv2.waitKey(1)  # 1초 대기
    if keyboard.is_pressed('b'):  # b를 누를 시 프로그램 종료
        break
f.close()