# YOLOv3 and Depth-map

1. Clone this repository:
```
git clone https://github.com/Abhinandan11/depth-map.git
```

2. Install requirements and download pretrained weights

```
$ pip3 install -r ./docs/requirements.txt
$ wget https://pjreddie.com/media/files/yolov3.weights


3. Download these two files: https://drive.google.com/open?id=1ztQTAoDOzfG9IiZc4rxB5FMJgi40ngvf
                            
   https://drive.google.com/open?id=1W7SexKT0Wb8-wqmf23q94tDLmd48wkxK
                             
   and move them to **models** folder.


4. Rename your input image as **xxx.jpg** and place it to **docs** folder.


5. Run **Reading_frame.py** file.

6. Run **YOLOv3_object_recognition.py** file. It will save results to **yoloimages**floder.

7. Run **predict.py** file with command: 
 
```
python predict.py models/NYU_FCRN.ckpt input.jpg
```


8. It will save two depth maps as **final.jpg** and **finalgray.jpg** to folders **final_depth** and **finalgray_depth**  respectively.


