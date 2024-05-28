import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import tensorflow as tf
from keras.models import load_model
import matplotlib.pyplot as plt

# Налаштування detectron2
detection_model_path = "D://University/diplom/models/model_final_detec.pth"  # замініть на ваш шлях
classification_model_path = "D://University/diplom/models/cnn_model1.h5"  # замініть на ваш шлях

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.MODEL.WEIGHTS = detection_model_path
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # змініть на вашу кількість класів
cfg.MODEL.DEVICE = "cpu"  # Використовуйте CPU замість GPU
predictor = DefaultPredictor(cfg)

# Змінна для збереження обрізаних зображень знаків
cropped_signs = []

# Розміри зображення
IMG_HEIGHT = 30
IMG_WIDTH = 30
channels = 3

# Словник класів
classes = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)', 3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)', 6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)', 9: 'No passing', 10: 'No passing veh over 3.5 tons', 11: 'Right-of-way at intersection',
    12: 'Priority road', 13: 'Yield', 14: 'Stop', 15: 'No vehicles', 16: 'Veh > 3.5 tons prohibited', 17: 'No entry',
    18: 'General caution', 19: 'Dangerous curve left', 20: 'Dangerous curve right', 21: 'Double curve',
    22: 'Bumpy road', 23: 'Slippery road', 24: 'Road narrows on the right', 25: 'Road work', 26: 'Traffic signals',
    27: 'Pedestrians', 28: 'Children crossing', 29: 'Bicycles crossing', 30: 'Beware of ice/snow',
    31: 'Wild animals crossing', 32: 'End speed + passing limits', 33: 'Turn right ahead', 34: 'Turn left ahead',
    35: 'Ahead only', 36: 'Go straight or right', 37: 'Go straight or left', 38: 'Keep right', 39: 'Keep left',
    40: 'Roundabout mandatory', 41: 'End of no passing', 42: 'End no passing veh > 3.5 tons'
}

def detect_and_crop(image_path):
    global cropped_signs
    # Завантаження зображення
    im = cv2.imread(image_path)
    height, width, _ = im.shape

    # Детекція знаків
    outputs = predictor(im)
    boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()

    # Обрізка кожного задетектованого знака
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int)

        # Розширення bounding box на 20%
        w = x2 - x1
        h = y2 - y1
        x1 = max(0, x1 - int(0.2 * w))
        y1 = max(0, y1 - int(0.2 * h))
        x2 = min(width, x2 + int(0.2 * w))
        y2 = min(height, y2 + int(0.2 * h))

        sign = im[y1:y2, x1:x2]
        sign = cv2.resize(sign, (IMG_WIDTH, IMG_HEIGHT))  # Зміна розміру
        cropped_signs.append((sign, (x1, y1, x2, y2)))  # Зберігання координат

    # Обведення рамкою всіх знаків після обрізання
    for _, (x1, y1, x2, y2) in cropped_signs:
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)

    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.show()

# Виклик функції на завантаженому зображенні
detect_and_crop('D://University/diplom/Test/test_photo/00626.jpg')

# Перевірка обрізаних знаків
print(f"Знайдено та обрізано {len(cropped_signs)} знаків.")
for i, (sign, _) in enumerate(cropped_signs):
    enlarged_sign = cv2.resize(sign, (IMG_WIDTH * 3, IMG_HEIGHT * 3))  # Збільшення зображення
    plt.imshow(cv2.cvtColor(enlarged_sign, cv2.COLOR_BGR2RGB))
    plt.show()

# Завантаження моделі класифікації
classification_model = load_model(classification_model_path)

# Функція для класифікації знака
def classify_sign(sign_image):
    # Додавання осі для відповідності очікуваному вхідному формату моделі
    sign_image = np.expand_dims(sign_image, axis=0)

    # Передбачення класу знака
    pred_class = classification_model.predict(sign_image)
    predicted_class = np.argmax(pred_class, axis=1)
    confidence = np.max(pred_class)  # Отримання відсотка впевненості

    return predicted_class[0], confidence

# Виклик функції для класифікації
for i, (sign, _) in enumerate(cropped_signs):
    predicted_class, confidence = classify_sign(sign)
    class_name = classes[predicted_class]  # Отримання назви класу
    print(f"Знак {i+1}: Передбачений клас: {class_name}, Впевненість: {confidence:.2f}")
    enlarged_sign = cv2.resize(sign, (IMG_WIDTH * 3, IMG_HEIGHT * 3))  # Збільшення зображення
    plt.imshow(cv2.cvtColor(enlarged_sign, cv2.COLOR_BGR2RGB))
    plt.title(f"Class: {class_name}, Confidence: {confidence:.2f}")
    plt.show()