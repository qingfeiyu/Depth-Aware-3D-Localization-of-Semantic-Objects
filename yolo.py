import cv2
import numpy as np
import core.utils as utils
import tensorflow as tf
from core.yolov3 import YOLOv3, decode
from PIL import Image
import matplotlib.pyplot as plt

import pandas as pd
import os, csv

def yolov3(file_path,file,output_path):
    input_size = 416
    # image_path   = "./docs/2.jpg"
    # image_path   = "./test.jpg"
    input_path = os.path.join(file_path, file)

    input_layer = tf.keras.layers.Input([input_size, input_size, 3])
    feature_maps = YOLOv3(input_layer)

    original_image = cv2.imread(input_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_size = original_image.shape[:2]

    image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    bbox_tensors = []
    for i, fm in enumerate(feature_maps):
        bbox_tensor = decode(fm, i)
        bbox_tensors.append(bbox_tensor)

    model = tf.keras.Model(input_layer, bbox_tensors)
    utils.load_weights(model, "./yolov3.weights")
    model.summary()

    pred_bbox = model.predict(image_data)
    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)
    bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.3)
    bboxes = utils.nms(bboxes, 0.45, method='nms')

    """
            bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
    """
    box_matrix = np.matrix(bboxes)

    box_size = np.zeros((len(box_matrix), 5))
    for i in range(0, len(box_matrix)):
        box_size[i, 0] = box_matrix[i, 0]
        box_size[i, 1] = box_matrix[i, 1]
        box_size[i, 2] = box_matrix[i, 2]
        box_size[i, 3] = box_matrix[i, 3]
        box_size[i, 4] = box_matrix[i, 5]
    name = ['x_min', 'y_min', 'x_max', 'y_max', 'cls_id']

    # make dictionary
    id_name = {}

    with open('./name.csv', 'r', encoding='utf-8') as F:
        reader = csv.reader(F)
        data = [a for a in reader]
        for i in range(len(data)):
            id_name[i] = data[i][0]

    class_id = list()
    # map id and names
    for i in range(0, len(box_size)):
        class_id.append(id_name[int(box_size[i, 4])])

    box_size_data = pd.DataFrame(columns=name, data=box_size)
    box_size_data['class_id'] = class_id
    print(box_size_data)

    box_size_data = box_size_data.drop(['cls_id'], axis=1)
    print(box_size_data)

    csv_path=r'./yolo_coordinates'
    # print(box_size_data)
    box_size_data.to_csv(os.path.join(csv_path,os.path.splitext(file)[0]+'.csv'), encoding='gbk', index=False)
    # print(bboxes[1])
    image = utils.draw_bbox(original_image, bboxes)

    image = Image.fromarray(image)
    # image.show()
    image.save(os.path.join(output_path, file))
    # print(image.width,image.height)


