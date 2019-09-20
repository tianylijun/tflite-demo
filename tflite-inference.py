# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""label_image for tflite."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import cv2

from tensorflow.lite.python.interpreter import Interpreter

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '-i',
      '--image',
      default='/tmp/grace_hopper.bmp',
      help='image to be classified')
  parser.add_argument(
      '-m',
      '--model_file',
      default='/tmp/mobilenet_v1_1.0_224_quant.tflite',
      help='.tflite model to be executed')
  parser.add_argument(
      '--input_mean',
      default=127.5, type=float,
      help='input_mean')
  parser.add_argument(
      '--input_std',
      default=0.0078125, type=float,
      help='input standard deviation')
  args = parser.parse_args()
  print(args.model_file)
  print(args.image)
  print(args.input_mean)
  print(args.input_std)
  interpreter = Interpreter(model_path=args.model_file)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  print(input_details)
  output_details = interpreter.get_output_details()
  print(output_details)
  # check the type of the input tensor
  floating_model = input_details[0]['dtype'] == np.float32

  # NxHxWxC, H:1, W:2
  height = input_details[0]['shape'][1]
  width = input_details[0]['shape'][2]
  #print(height)
  #print(width)
  img = cv2.imread(args.image, 0)
  #print(img.shape[0])
  #print(img.shape[1])
  
  #imgs=[]
  #imgs.append(img)
  #imgs.append(img)
  #imgs = np.array(imgs)
  #print(imgs.shape)
  
  # add N dim
  input_data = np.expand_dims(img, axis=2)
  input_data = np.expand_dims(input_data, axis=0)
  print(input_data.shape)
  print("===========================")

  start = time.time()

  if floating_model:
    input_data = (np.float32(input_data) - args.input_mean) * args.input_std
  interpreter.set_tensor(input_details[0]['index'], input_data)
  #for num in range(10):
  interpreter.invoke()
  output_data = interpreter.get_tensor(output_details[0]['index'])

  stop = time.time()
  print("time(s):", stop-start)
  #results = np.squeeze(output_data)
  #print(results)
