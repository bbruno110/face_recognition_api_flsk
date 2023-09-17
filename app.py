from flask import Flask, request, jsonify, send_file  # Importe a função send_file
import dlib
import cv2
import logging
import os
import base64
import numpy as np
import logging
from flask_cors import CORS
import uuid  # Importe o módulo uuid

app = Flask(__name__)
CORS(app)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'
logging.basicConfig(filename='flask_app.log', level=logging.DEBUG)
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
    if 'photo' not in request.files:
        return jsonify({'message': 'Nenhuma imagem enviada'}), 400

    photo = request.files['photo']

    if not allowed_file(photo.filename):
        return jsonify({'message': 'Extensão de arquivo não suportada'}), 400

    img_bytes = photo.read()
    img_np = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray, 1)

    if len(faces) > 0:
        for face in faces:
            shape = shape_predictor(gray, face)
            top_lip = shape.part(62).y
            bottom_lip = shape.part(66).y

            # Calcule o ponto médio entre os lábios
            lip_midpoint = (top_lip + bottom_lip) // 2

            # Defina um limiar para determinar se os dentes estão visíveis ou não
            lip_distance_threshold = 20  # Ajuste este valor conforme necessário

            teeth_points = shape.parts()[60:68]
            teeth_points = np.array([[p.x, p.y] for p in teeth_points])
            
            # Calcule a distância entre o ponto médio dos lábios e o lábio inferior
            lip_distance = abs(bottom_lip - lip_midpoint)

            if lip_distance > lip_distance_threshold:
                cv2.drawContours(img, [teeth_points], -1, (0, 255, 0), 2)
            else:
                return jsonify({'message': 'Nenhum dente encontrado'}), 206

        unique_filename = str(uuid.uuid4()) + '.jpg'
        photo_path = os.path.join(img_folder, unique_filename)
        cv2.imwrite(photo_path, img)

        with open(photo_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

        # Remova a imagem após enviá-la em base64
        os.remove(photo_path)

        return jsonify({'image': encoded_image}), 200
    else:
        return jsonify({'message': 'Nenhum rosto detectado na imagem'}), 204

@app.route('/convert_to_jpg', methods=['POST'])
def convert_to_jpg():
    try:
        data = request.json  # Assume que a imagem em base64 é enviada como um objeto JSON com a chave 'base64_image'

        if 'base64_image' not in data:
            return jsonify({'message': 'Base64 image not found in request'}), 400

        base64_image = data['base64_image']

        # Verifique se a string base64 começa com o cabeçalho "data:image/jpeg;base64,"
        if base64_image.startswith("data:image/jpeg;base64,"):
            # Remova o cabeçalho para obter apenas os dados de imagem em base64
            base64_image = base64_image[len("data:image/jpeg;base64,"):]

        # Decodificar a imagem em base64
        image_data = base64.b64decode(base64_image)

        # Verifique se image_data não está vazio
        if not image_data:
            return jsonify({'message': 'Invalid base64 image data'}), 400

        # Gerar um nome de arquivo único
        unique_filename = str(uuid.uuid4()) + '.jpeg'

        # Defina o caminho para salvar a imagem jpg na pasta "imgs"
        jpg_filename = os.path.join(img_folder, unique_filename)

        # Salve a imagem como jpg
        with open(jpg_filename, 'wb') as img_file:
            img_file.write(image_data)

        # Retorne a imagem como parte da resposta
        return send_file(jpg_filename, mimetype='image/jpeg')

    except Exception as e:
        # Registre a exceção no arquivo de log
        app.logger.exception("Erro durante o processamento da requisição:")

        # Retornar uma resposta de erro (status 500) em JSON
        return jsonify({'error': str(e)}), 500

   
@app.route('/', methods=['GET'])
def ping():
    return jsonify({'message': 'pong'})

if __name__ == '__main__':
    
      app.run(host='0.0.0.0', port=5000)
