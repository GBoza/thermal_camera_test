# main.py

# Standard library imports
import time
import csv
from datetime import datetime

# Third party imports
from flask import Flask, render_template, Response, json

# Local application imports
from libs.htpa import *

import cv2
import base64
from flask import make_response
import numpy as np

app = Flask(__name__)

cam = cv2.VideoCapture(0)

# Thermal camera driver
dev = HTPA(0x1A)

def remap(x,a,b,c,d):
    r = c + ((x - a)*(d - c)/(b - a))
    return r
    
@app.route('/camera')
def camera():
    ret, img = cam.read()
    if ret:
        img = cv2.resize(img, (320, 200), interpolation = cv2.INTER_CUBIC)
        ret, buffer = cv2.imencode('.png', img)
        response = make_response(buffer.tobytes())
        response.headers['Content-Type'] = 'image/png'
        return response
    else:
        return "FAIL"

@app.route('/thermal-camera')
def thermal():
    #fr = dev.get_frame()
    temp, fr = dev.get_frame_temperature()
    #norm_image = cv2.normalize(fr, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    #norm_image = np.multiply(fr, 0.1)
    remap_fr = remap(fr, fr.min(), 50, 0, 255) #remap
    remap_fr = np.round(remap_fr) #round
    remap_fr = remap_fr.astype(np.uint8) #as 8bit
    imgthermal = cv2.applyColorMap(remap_fr, cv2.COLORMAP_HOT) #colormap
    imgthermal = cv2.resize(imgthermal, (256, 256), interpolation = cv2.INTER_CUBIC) #interpolation
    ret, buffer = cv2.imencode('.png', imgthermal) #encode
    response = make_response(buffer.tobytes())
    response.headers['Content-Type'] = 'image/png'
    return response

@app.route('/home')
def home():
    return "<h1>Thermal measure HTPA - Visiontech</h1>"

@app.route('/raw-data', methods=['GET'])
# 
def get_raw_data():
    fr = dev.get_frame()
    frame = fr.tolist()
    temp = 25.5 

    # time stamp 
    now = datetime.now() # current date and time
    date_time = now.strftime("%m/%d/%Y %H:%M:%S")

    data = {"raw-data" : frame, "ts": date_time}
    
    response = app.response_class(
        response = json.dumps(data),
        status = 200,
        mimetype = 'application/json'
    )
    return response

@app.route('/temperature', methods=['GET'])
# 
def get_raw_temperature():
    temp, fr = dev.get_frame_temperature()
    frame = fr.tolist() 

    # time stamp 
    now = datetime.now() # current date and time
    date_time = now.strftime("%m/%d/%Y %H:%M:%S")

    data = {"raw-temperature" : frame, "amb-temp" : temp,
            "ts": date_time}
    
    response = app.response_class(
        response = json.dumps(data),
        status = 200,
        mimetype = 'application/json'
    )
    return response

if __name__ == '__main__':
    # defining server ip address and port
    app.run(host='0.0.0.0',port='5050', debug=False)
