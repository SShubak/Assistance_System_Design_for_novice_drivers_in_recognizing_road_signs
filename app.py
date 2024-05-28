from flask import Flask, request, render_template, url_for
import os
import cv2
import numpy as np
from keras.models import load_model
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'D://University/diplom/static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Завантаження моделей
detection_model_path = "D://University/diplom/models/model_final_detec.pth"  # замініть на ваш шлях
classification_model_path = "D://University/diplom/models/cnn_model1.h5"  # замініть на ваш шлях

classification_model = load_model(classification_model_path)

classes = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing veh over 3.5 tons',
    11: 'Right-of-way at intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Veh > 3.5 tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve left',
    20: 'Dangerous curve right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End speed + passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End no passing veh > 3.5 tons'
}

recommendations = {
    'Speed limit (20km/h)': 'Зменште швидкість до 20 км/год.',
    'Speed limit (30km/h)': 'Зменште швидкість до 30 км/год.',
    'Speed limit (50km/h)': 'Зменште швидкість до 50 км/год.',
    'Speed limit (60km/h)': 'Зменште швидкість до 60 км/год.',
    'Speed limit (70km/h)': 'Зменште швидкість до 70 км/год.',
    'Speed limit (80km/h)': 'Зменште швидкість до 80 км/год.',
    'End of speed limit (80km/h)': 'Обмеження швидкості 80 км/год закінчується.',
    'Speed limit (100km/h)': 'Зменште швидкість до 100 км/год.',
    'Speed limit (120km/h)': 'Зменште швидкість до 120 км/год.',
    'No passing': 'Обгін заборонено.',
    'No passing veh over 3.5 tons': 'Обгін заборонено для транспортних засобів понад 3.5 тонни.',
    'Right-of-way at intersection': 'Поступіться дорогою на перехресті.',
    'Priority road': 'Головна дорога.',
    'Yield': 'Дайте дорогу.',
    'Stop': 'Зупиніться.',
    'No vehicles': 'Рух заборонено.',
    'Veh > 3.5 tons prohibited': 'Рух заборонено для транспортних засобів понад 3.5 тонни.',
    'No entry': 'В’їзд заборонено.',
    'General caution': 'Загальна обережність.',
    'Dangerous curve left': 'Небезпечний поворот ліворуч.',
    'Dangerous curve right': 'Небезпечний поворот праворуч.',
    'Double curve': 'Подвійний поворот.',
    'Bumpy road': 'Нерівна дорога.',
    'Slippery road': 'Слизька дорога.',
    'Road narrows on the right': 'Дорога звужується праворуч.',
    'Road work': 'Дорожні роботи. Будьте уважні!',
    'Traffic signals': 'Світлофор.',
    'Pedestrians': 'Пішоходи.',
    'Children crossing': 'Діти переходять дорогу.',
    'Bicycles crossing': 'Перехід велосипедистів.',
    'Beware of ice/snow': 'Обережно: лід/сніг.',
    'Wild animals crossing': 'Перехід диких тварин.',
    'End speed + passing limits': 'Кінець обмежень швидкості та обгону.',
    'Turn right ahead': 'Поверніть праворуч попереду.',
    'Turn left ahead': 'Поверніть ліворуч попереду.',
    'Ahead only': 'Рух тільки прямо.',
    'Go straight or right': 'Рух тільки прямо або праворуч.',
    'Go straight or left': 'Рух тільки прямо або ліворуч.',
    'Keep right': 'Тримайтесь праворуч.',
    'Keep left': 'Тримайтесь ліворуч.',
    'Roundabout mandatory': 'Круговий рух обов’язковий.',
    'End of no passing': 'Кінець заборони обгону.',
    'End no passing veh > 3.5 tons': 'Кінець заборони обгону для транспортних засобів понад 3.5 тонни.'
}

# Налаштування detectron2
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.MODEL.WEIGHTS = detection_model_path
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # змініть на вашу кількість класів
cfg.MODEL.DEVICE = "cpu"  # Використовуйте CPU замість GPU
predictor = DefaultPredictor(cfg)

IMG_HEIGHT = 30
IMG_WIDTH = 30
channels = 3


def detect_and_crop(image_path):
    im = cv2.imread(image_path)
    height, width, _ = im.shape
    outputs = predictor(im)
    boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
    cropped_signs = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int)
        w = x2 - x1
        h = y2 - y1
        x1 = max(0, x1 - int(0.2 * w))
        y1 = max(0, y1 - int(0.2 * h))
        x2 = min(width, x2 + int(0.2 * w))
        y2 = min(height, y2 + int(0.2 * h))
        sign = im[y1:y2, x1:x2]
        sign = cv2.resize(sign, (30, 30))
        cropped_signs.append((sign, (x1, y1, x2, y2)))
    for _, (x1, y1, x2, y2) in cropped_signs:
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
    detected_image_path = os.path.join(UPLOAD_FOLDER, os.path.basename(image_path).split('.')[0] + "_detected.jpg")
    cv2.imwrite(detected_image_path, im)
    return cropped_signs, detected_image_path

def classify_sign(sign_image):
    sign_image = np.expand_dims(sign_image, axis=0)
    pred_class = classification_model.predict(sign_image)
    predicted_class = np.argmax(pred_class, axis=1)
    confidence = np.max(pred_class)
    return classes[predicted_class[0]], confidence


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        cropped_signs, detected_image_path = detect_and_crop(file_path)
        results = []
        for i, (sign, _) in enumerate(cropped_signs):
            predicted_class, confidence = classify_sign(sign)
            sign_filename = os.path.basename(file_path).split('.')[0] + f"_sign_{i}.jpg"
            sign_path = os.path.join(app.config['UPLOAD_FOLDER'], sign_filename)
            cv2.imwrite(sign_path, sign)
            results.append((predicted_class, confidence, sign_filename, recommendations[predicted_class]))
        return render_template('index.html', filename=os.path.basename(detected_image_path), results=results)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)