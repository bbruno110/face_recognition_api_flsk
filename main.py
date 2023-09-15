from flask import Flask, request, jsonify, send_file  # Importe a função send_file
import dlib
import cv2
import logging
import os
import numpy as np
import time
from flask_cors import CORS, cross_origin
import uuid  # Importe o módulo uuid

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
logging.basicConfig(level=logging.DEBUG)
logger = app.logger
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.DEBUG)
app.logger.addHandler(file_handler)

# Carregue seus modelos e descritores faciais aqui
face_detector = dlib.get_frontal_face_detector()
face_recognizer = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Pasta onde as imagens serão armazenadas
img_folder = "imgs"

# Verifica se a pasta de imagens existe, senão cria
if not os.path.exists(img_folder):
    os.makedirs(img_folder)

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/detect_teeth', methods=['POST'])
def detect_teeth():
    logger.debug('Acessando a rota principal')
    # Verifique se a solicitação possui uma imagem
    if 'photo' not in request.files:
        return jsonify({'message': 'Nenhuma imagem enviada'}), 400

    photo = request.files['photo']

    # Verifique se a extensão da imagem é válida
    if not allowed_file(photo.filename):
        return jsonify({'message': 'Extensão de arquivo não suportada'}), 400

    # Gere um nome de arquivo único usando um UUID
    unique_filename = str(uuid.uuid4()) + '.jpg'
    photo_path = os.path.join(img_folder, unique_filename)
    photo.save(photo_path)

    # Carregue a imagem e execute a detecção de dentes
    img = cv2.imread(photo_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray, 1)

    if len(faces) > 0:
        for face in faces:
            # Encontre os pontos de referência do rosto
            shape = shape_predictor(gray, face)

            # Calcule a distância entre os pontos de referência dos lábios superior e inferior
            top_lip = shape.part(62).y
            bottom_lip = shape.part(66).y
            lip_distance = bottom_lip - top_lip

            # Defina um limiar para determinar se os dentes estão visíveis ou não
            lip_distance_threshold = 1

            # Desenhe um contorno verde ao redor dos dentes se eles estiverem visíveis, caso contrário, retorne uma mensagem
            teeth_points = shape.parts()[60:68]
            teeth_points = np.array([[p.x, p.y] for p in teeth_points])
            if lip_distance > lip_distance_threshold:
                cv2.drawContours(img, [teeth_points], -1, (0, 255, 0), 2)
            else:
                return jsonify({'message': 'Nenhum dente encontrado'}), 200  # Retorna a mensagem quando os dentes não são visíveis

        # Salve a imagem com os contornos dos dentes desenhados
        cv2.imwrite(photo_path, img)

        # Retorna a imagem resultante com o nome original como resposta
        return send_file(photo_path, mimetype='image/jpeg'), 200
    else:
        return jsonify({'message': 'Nenhum rosto detectado na imagem'}), 400

@app.route('/', methods=['GET'])
def ping():
    logger.debug('Acessando a rota principal')
    return jsonify({'message': 'pong'})

if __name__ == '__main__':
    app.run(debug=True)
