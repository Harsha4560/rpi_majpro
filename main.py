from tflite_runtime.interpreter import Interpreter
import numpy as np
import cv2
import pandas as pd
import os
from joblib import load
import matplotlib
import matplotlib.pyplot as plt
import time
plt.ion()

scaler = load('std_scaler.bin')
model_path = 'cnn_model.tflite'
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']




rtsp_url = 'rtsp://admin:Asdfghjkl@192.168.1.29:554/Streaming/Channels/101/'
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print('Camera not opened')
    exit()
    
font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 50)
font_scale = 1
font_color = (255, 255, 255)
line_type = 2

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print('Frame not read')
        break
    
    if frame_count % 10 == 0:
        print('Frame: ', frame_count)
        img = cv2.resize(frame, (224, 224))
        img = np.reshape(img, (1, 224, 224, 3))
        img = img / 255.0
        img = np.array(img, dtype=np.float32)
        interpreter.set_tensor(input_index, img)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_index)[0][0]
        prediction = np.reshape(prediction, (-1, 1))
        prediction = scaler.inverse_transform(prediction)[0][0]
        print('Prediction: ', prediction)
        cv2.putText(frame, 'GHI: ' + str(round(prediction, 2)), bottomLeftCornerOfText, font, font_scale, font_color, line_type)
        cv2.imshow('RTSP Stream', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
    frame_count += 1

cap.release()
cv2.destroyAllWindows()
