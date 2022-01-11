"""
2021 Eirene Solutions
Computer vision team
"""
# Import packages
import os
import argparse
# import cv2
import numpy as np
import time
from PIL import Image

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='model.tflite path',
                    required=True)
parser.add_argument('--image', help='Path of image to be inferenced',
                    default=None)
parser.add_argument('--img_size', help='Image size for model',
                    default=(224,224))

args = parser.parse_args()

MODEL = args.model
IMAGE_PATH = args.image
IMG_SIZE= args.img_size #not working xD
# print(IMG_SIZE)
#------------------------------------------------------------------------------
#Load TFlite Model  
import tflite_runtime.interpreter as tflite

# Load the TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path=MODEL)
interpreter.allocate_tensors()
model= interpreter

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

#Inference
input_data = Image.open(IMAGE_PATH) 
input_data= input_data.resize(IMG_SIZE, Image.ANTIALIAS)
input_data= np.array(input_data, dtype=np.uint8).reshape(-1,IMG_SIZE[0],IMG_SIZE[1],3)
interpreter.set_tensor(input_details[0]['index'], input_data)

start_time = time.monotonic()
interpreter.invoke()
elapsed_ms = (time.monotonic() - start_time) * 1000

print(f'Inference Time: {np.round_(elapsed_ms,2)} ms')

output_data = interpreter.get_tensor(output_details[0]['index'])

print("Sem daninha" if int(output_data) > 127 else "Com daninha")
#---------------------------------------------------------------

#functions

#Example
#python3 tflite_detect_script.py --model weeds.tflite --image img_1597293167.46.png