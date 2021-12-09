
# import system module
import sys

# import some PyQt5 modules
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer
import PyQt5.QtCore as QtCore
from PyQt5.QtWidgets import QMessageBox

# import Opencv module
import cv2
import libs.camera as face

import requests
import numpy as np
import json

import scipy.ndimage.filters as filters


from ui_main_window import *

def remap(x,a,b,c,d):
    r = c + ((x - a)*(d - c)/(b - a))
    return r

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized
    
def crop_image_from_center(image, h, w):
    center = image.shape[0] / 2 , image.shape[1] / 2
    x = center[1] - w / 2
    y = center[0] - h / 2
    crop_img = image[int(y):int(y+h), int(x):int(x+h)]
    return crop_img

class MainWindow(QWidget):
    # class constructor
    def __init__(self):
        # call QWidget constructor
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        # create a timer
        self.timer = QTimer()
        # set timer timeout callback function
        self.timer.timeout.connect(self.viewCam)
        # set control_bt callback clicked  function
        self.ui.control_bt.clicked.connect(self.controlTimer)

        
        self.stage = 0
        self.facesimg = list()

    # view camera
    def viewCam(self):
        # read image in BGR format
        #ret, image = self.cap.read()
        #if ret is False:
        #    return
        ##image = cv2.resize(image, (320, 240), interpolation = cv2.INTER_LINEAR)
        #image = image_resize(image, height = 256);
        #image = crop_image_from_center(image, 256, 256);
            

        #get image from flask api
        image_data = requests.get("http://localhost:5050/camera").content
        image =  np.asarray(bytearray(image_data), dtype ="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = image_resize(image, height = 256);
        image = crop_image_from_center(image, 256, 256);
        ## convert image to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        content_data = requests.get("http://localhost:5050/temperature").content
        data = json.loads(content_data)
        raw_temperature = data['raw-temperature']
        
        frame =  np.asarray(raw_temperature, dtype ="float")
        kernel_size = 4
        frame = filters.uniform_filter(frame, size=kernel_size, mode='constant')
        weights = filters.uniform_filter(np.ones(frame.shape), size=kernel_size, mode='constant')
        frame = frame / weights #normalized convolution result
        print("MaxTemp:",frame.max())
        self.ui.label_2.setText("Temp: " + str(round(frame.max(),2)) + "°C")
        print(frame)
        
        #thermal image
        thermal_img = remap(frame, frame.min(), 50, 0, 255) #remap
        thermal_img = np.round(thermal_img) #round
        thermal_img = thermal_img.astype(np.uint8) #as 8bit
        
        thermal_img = cv2.applyColorMap(thermal_img, cv2.COLORMAP_HOT) #colormap
        thermal_img = cv2.resize(thermal_img, (256, 256), interpolation = cv2.INTER_CUBIC) #resize / interporlation
        thermal_img = cv2.cvtColor(thermal_img, cv2.COLOR_BGR2RGB)
        #print(thermal_img.shape)


        orig_image = image.copy()
        
        face_rects = face.detector(image, 0)
        
        if len(face_rects) > 0:
            for face_rect in face_rects:
                image = cv2.rectangle(image, (face_rect.left(),face_rect.top()), (face_rect.right(),face_rect.bottom()),(255,0,0),2)
                          
                #crop image  
                crop_thermal = frame[int(face_rect.top()/8):int(face_rect.bottom()/8), int(face_rect.left()/8):int(face_rect.right()/8)]
                max_temp = round(crop_thermal.max(),1)
                image = cv2.putText(image, str(max_temp), (face_rect.left(), face_rect.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                print("FaceTemp:", str(max_temp) +"°C")
        
        # get image infos
        height, width, channel = image.shape
        step = channel * width
        # create QImage from image
        qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
        # show image in img_label
        self.ui.image_label.setPixmap(QPixmap.fromImage(qImg))
        
        # get image infos
        height, width, channel = thermal_img.shape
        step = channel * width
        # create QImage from image
        qImg = QImage(thermal_img.data, width, height, step, QImage.Format_RGB888)
        # show image in img_label
        self.ui.label.setPixmap(QPixmap.fromImage(qImg))

    # start/stop timer
    def controlTimer(self):
        # if timer is stopped
        if not self.timer.isActive():
            # create video capture
            self.cap = cv2.VideoCapture(0)
            # start timer
            self.timer.start(20)
            # update control_bt text
            self.ui.control_bt.setText("Stop")
        # if timer is started
        else:
            # stop timer
            self.timer.stop()
            # release video capture
            self.cap.release()
            # update control_bt text
            self.ui.control_bt.setText("Start")
            


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # create and show mainWindow
    mainWindow = MainWindow()
    mainWindow.show()

    sys.exit(app.exec_())
