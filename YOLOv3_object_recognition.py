import os
import yolo

# 生成图片的路径
output_path = r'./yoloimages'

# 生成img_path文件夹
if not os.path.isdir(output_path):
    os.mkdir(output_path)



yolo.yolov3('./images','./000000.jpg',output_path)
#yolo.yolo('./images','./000001.jpg',output_path)
#yolo.yolo('./images','./000002.jpg',output_path)
#yolo.yolo('./images','./000003.jpg',output_path)
##yolo.yolo('./images',file,output_path)