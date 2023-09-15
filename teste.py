import dlib
import cv2
import os
import numpy as np
import time

face_detector = dlib.get_frontal_face_detector()
face_recognizer = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

cur_direc = os.getcwd()
path = os.path.join(cur_direc + '/img/')
folder_path = path
face_descriptors = []

# Variáveis para calcular o FPS
start_time = time.time()
frame_count = 0

for filename in os.listdir(folder_path):
    img = cv2.imread(os.path.join(folder_path, filename))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detecte o rosto na imagem
    faces = face_detector(gray, 1)
    if len(faces) > 0:
        # Encontre os pontos de referência do rosto
        shape = shape_predictor(gray, faces[0])
        # Calcule o descritor facial da imagem
        face_descriptor = face_recognizer.compute_face_descriptor(img, shape)
        face_descriptors.append(face_descriptor)

# Converta a lista de descritores faciais em um array numpy
face_descriptors = np.array(face_descriptors)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray, 1)

    # Atualizar o contador de quadros
    frame_count += 1
    elapsed_time = time.time() - start_time

    # Calcular o FPS a cada segundo
    if elapsed_time > 1.0:
        fps = frame_count / elapsed_time
        start_time = time.time()
        frame_count = 0

    # Exibir o FPS no canto superior esquerdo do quadro
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        # Encontre os pontos de referência do rosto
        shape = shape_predictor(frame, face)

        # Calcule a distância entre os pontos de referência dos lábios superior e inferior
        top_lip = shape.part(62).y
        bottom_lip = shape.part(66).y
        lip_distance = bottom_lip - top_lip

        # Defina um limiar para determinar se os dentes estão visíveis ou não
        lip_distance_threshold = 1

        # Desenhe um contorno verde ao redor dos dentes se eles estiverem visíveis, caso contrário desenhe um contorno vermelho
        teeth_points = shape.parts()[60:68]
        teeth_points = np.array([[p.x, p.y] for p in teeth_points])
        if lip_distance > lip_distance_threshold:
            cv2.drawContours(frame, [teeth_points], -1, (0, 255, 0), 2)
        else:
            cv2.drawContours(frame, [teeth_points], -1, (0, 0, 255), 2)

        # Calcule o descritor facial do quadro
        face_descriptor = face_recognizer.compute_face_descriptor(frame, shape)

        # Compare o descritor facial com os descritores das imagens na pasta "img"
        distances = np.linalg.norm(face_descriptors - face_descriptor, axis=1)
        best_match_index = np.argmin(distances)

        # Se a distância for pequena o suficiente (abaixo de um limiar), exiba o nome do arquivo da imagem correspondente
        if distances[best_match_index] < 0.48:
            matched_image = os.listdir(folder_path)[best_match_index]
            filename_without_extension = os.path.splitext(matched_image)[0]
            cv2.putText(frame, filename_without_extension, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Exiba o frame com o contorno desenhado ao redor da boca e a mensagem de correspondência
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libere a captura de vídeo e feche a janela
cap.release()
cv2.destroyAllWindows()
