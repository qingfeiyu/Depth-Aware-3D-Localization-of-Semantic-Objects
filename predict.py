import argparse
import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import models
import pandas as pd
import math

def MaxMinNormalization(x,Max,Min):
    x = (x - Min) / (Max - Min)
    return x


def predict(model_data_path, image_path):

    # yolo
    #dection_size = image_demo.image_yolo(image_path)
    #print(dection_size)

    # Default input size
    height = 240
    width = 320
    #height = 360
    #width = 480
    channels = 3
    batch_size = 1
   
    # Read image
    img = Image.open(image_path)
    #img=img.transpose(Image.ROTATE_270)
    print(img.size)
    img = img.resize([width,height], Image.ANTIALIAS)
    img = np.array(img).astype('float32')
    img = np.expand_dims(np.asarray(img), axis = 0)
    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))

    # Construct the network
    net = models.ResNet50UpProj({'data': input_node}, batch_size, 1, False)
        
    with tf.Session() as sess:

        # Load the converted parameters
        print('Loading the model')

        # Use to load from ckpt file
        saver = tf.train.Saver()     
        saver.restore(sess, model_data_path)

        # Use to load from npy file
        #net.load(model_data_path, sess) 

        # Evalute the network for the given image
        pred = sess.run(net.get_output(), feed_dict={input_node: img})
        print(pred[0, :, :, 0])
        #print(pred[0,0,0,0])
        print(len(pred[0,:,0, 0]))
        print(len(pred[0, 0, :, 0]))
        #print(pred[0, 351,527, 0])

        # yolo
        """
            yolo_result: [x_min, y_min, x_max, y_max, cls_id] format coordinates.
        """
        original_yolo_result = pd.read_csv('./yolo_coordinates/000000.csv', index_col=None)
        yolo_result = original_yolo_result.values.tolist()

        #print(yolo_result)
        #for i in range(0, len(yolo_result)):
            #yolo_result[i][0] = math.floor((len(pred[0, 0, :, 0]) / 1280) * (yolo_result[i][0]))  # x_min
            #yolo_result[i][2] = math.floor((len(pred[0, 0, :, 0]) / 1280) * (yolo_result[i][2]))  # x_max
            #yolo_result[i][1] = math.floor((len(pred[0,:,0, 0]) / 720) * (yolo_result[i][1]))  # y_min
            #yolo_result[i][3] = math.floor((len(pred[0,:,0, 0]) / 720) * (yolo_result[i][3]))  # y_max
        #print(yolo_result)
        for i in range(0, len(yolo_result)):
            yolo_result[i][0] = math.floor((len(pred[0, 0, :, 0]) / 640) * (yolo_result[i][0]))  # x_min
            yolo_result[i][2] = math.floor((len(pred[0, 0, :, 0]) / 640) * (yolo_result[i][2]))  # x_max
            yolo_result[i][1] = math.floor((len(pred[0,:,0, 0]) / 480) * (yolo_result[i][1]))  # y_min
            yolo_result[i][3] = math.floor((len(pred[0,:,0, 0]) / 480) * (yolo_result[i][3]))  # y_max

        # calculate average distance

        average_distance_list=[]
        average_distance=0
        point_size=0
        for i in range(0,len(yolo_result)):
            for w in range(yolo_result[i][0],yolo_result[i][2]):
                for h in range(yolo_result[i][1],yolo_result[i][3]):
                    average_distance=average_distance+pred[0,h,w,0]
                    point_size=point_size+1
            avg_distance=average_distance/point_size
            average_distance_list.append(round(avg_distance,3))
            print("object:",yolo_result[i][4],"average distance is: ",round(avg_distance,3))
        #print(average_distance_list)
        original_yolo_result['average_distance'] = average_distance_list
        #print(original_yolo_result)

        # calculate center distance
        center_distance_list = []
        center_distance = 0

        for i in range(0, len(yolo_result)):
            center_x = round((yolo_result[i][2] + yolo_result[i][0])/2)
            center_y = round((yolo_result[i][3] + yolo_result[i][1])/2)
            center_distance=pred[0,center_y,center_x,0]
            center_distance_list.append(round(center_distance, 3))
            print("object:", yolo_result[i][4], "center distance is: ", round(center_distance, 3))
        #print(center_distance_list)
        original_yolo_result['center_distance'] = center_distance_list
        #print(original_yolo_result)

        # calculate weighted average distance

        weighted_average_distance_list = []
        weighted_average_distance = 0
        denominator=0
        for i in range(0, len(yolo_result)):
            center_x = round((yolo_result[i][2] + yolo_result[i][0]) / 2)
            center_y = round((yolo_result[i][3] + yolo_result[i][1]) / 2)
            sum_distance = 0
            s=0

            x_d = (yolo_result[i][2] - center_x) ** 2
            y_d = (yolo_result[i][3] - center_y) ** 2
            distance_d = math.sqrt(x_d+y_d)
            #print(distance_d)

            for w in range(yolo_result[i][0], yolo_result[i][2]):
                for h in range(yolo_result[i][1], yolo_result[i][3]):
                    x=(w-center_x)**2
                    y=(h-center_y)**2
                    r = math.sqrt(x+y)
                    # 归一化
                    #r=MaxMinNormalization(r,distance_d,0)

                    s=math.exp(-(r**2))
                    weight_distance=s * pred[0, h, w, 0]
                    sum_distance = sum_distance + weight_distance
                    denominator =  denominator + s
            weighted_average_distance = sum_distance / denominator
            weighted_average_distance_list.append(round(weighted_average_distance, 3))
            print("object:", yolo_result[i][4], "weighted_average distance is: ", round(weighted_average_distance, 3))
        #print(weighted_average_distance_list)
        original_yolo_result['weighted_average_distance'] = weighted_average_distance_list
        print(original_yolo_result)



        csv_path = r'./depth_result'
        # print(box_size_data)
        original_yolo_result.to_csv(os.path.join(csv_path, './000000.csv'), encoding='gbk', index=False)

        # Plot result
        #fig = plt.figure(image_path)
        fig = plt.figure()
        ii = plt.imshow(pred[0,:,:,0], interpolation='nearest')
        fig.colorbar(ii)

    plt.savefig(os.path.join(r'./final_depth', './000000.jpg'))
    image = cv2.imread(os.path.join(r'./final_depth', './000000.jpg'))
    graysc = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(r'./finalgray_depth', './000000.jpg'), graysc)


    #plt.savefig('final.jpg')
    #image = cv2.imread('final.jpg')
    #graysc = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #cv2.imwrite('finalgray.jpg', graysc)

    return pred
        
                
def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Converted parameters for the model')
    parser.add_argument('image_paths', help='Directory of images to predict')
    args = parser.parse_args()

    # Predict the image
    pred = predict(args.model_path, args.image_paths)

    #predict(models/NYU_FCRN.ckpt,input.jpg)
    os._exit(0)

if __name__ == '__main__':
    main()

        



