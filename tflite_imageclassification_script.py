# -*- coding: utf-8 -*-
"""tflite_detect_script.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15DXKtEf_ts3VTZ2waTYDzi1AfyOcyxWR
"""

# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import glob
import importlib.util

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='model.tflite path',
                    required=True)
parser.add_argument('--image', help='Path of image to be inferenced',
                    default=None)
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.3)

args = parser.parse_args()

MODEL = args.model
IMAGE_PATH = args.image
TRESH= args.threshold
#------------------------------------------------------------------------------
#Load TFlite Model  
import numpy as np
import tflite_runtime.interpreter as tflite

# Load the TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path=MODEL)
interpreter.allocate_tensors()
model= interpreter

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.uint8)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
#---------------------------------------------------------------

#functions
import cv2
from PIL import Image
import time 

def resize_img_preprocess(image_path, input_size):
  image = cv2.imread(image_path)
  image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  imH, imW, _ = image.shape 
  image_resized = cv2.resize(image_rgb, input_size)
  input_data = np.expand_dims(image_resized, axis=0)
  return image_resized


def preprocess_image(image_path, input_size):
  """Preprocess the input image to feed to the TFLite model"""
  img = cv2.imread(image_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  original_image = img
  resized_img= resize_img_preprocess(image_path, input_size)
  return resized_img, original_image


def set_input_tensor(interpreter, image):
  """Set the input tensor."""
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
  """Retur the output tensor at the given index."""
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor


def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""
  # Feed the input image to the model
  set_input_tensor(interpreter, image)

  start_time = time.monotonic()
  interpreter.invoke()
  elapsed_ms = (time.monotonic() - start_time) * 1000
  print(f'Inference Time: {np.round_(elapsed_ms,2)} ms')

  # Get all outputs from the model
  results = get_output_tensor(interpreter, 0)
  return results


def run_odt_and_draw_results(image_path, interpreter, threshold=0.5):
  """Run object detection on the input image and draw the detection results"""
  # Load the input shape required by the model
  _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

  # Load the input image and preprocess it
  preprocessed_image, original_image = preprocess_image(
      image_path,
      (input_height, input_width)
    )

  # Run object detection on the input image
  results = detect_objects(interpreter, preprocessed_image, threshold=threshold)

  return results
#------------------------------------------------------------------------------
#inference

model_path= MODEL
detection_treshold = TRESH
img_path= IMAGE_PATH

# Load the TFLite model
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Run inference and draw detection result on the local copy of the original file
detection_result_image = run_odt_and_draw_results(
    img_path,
    interpreter,
    threshold=detection_treshold
)

print(detection_result_image)

#Example
#python3 tflite_detect_script.py --model weeds.tflite --image img_1597293167.46.png