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

import os
import sys
import time
import argparse
import numpy as np
import cv2

from tensorflow.lite.python.interpreter import Interpreter

def mkdir(path):
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False

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
  img_list=[]
  with open(args.image, 'r') as f:
      for line in f:
          img_list.append(line.strip())
  #print(img_list)
  i = 0
  for img_path in img_list:
    print(img_path)
    out_img_path = img_path.replace("/Users/leejohnnie/nfs/lfw-matlab-112x112", "./result")
    #out_img_path = img_path.replace("/Users/leejohnnie/dataset", "./result")
    print(out_img_path)
    out_img_path = os.path.splitext(out_img_path)
    print(out_img_path)
    newname = out_img_path[0] + ".bin"
    print(newname)
    mkdir(os.path.dirname(newname))
    start = time.time()
    img = cv2.imread(img_path, 0)
    input_data = np.expand_dims(img, axis=2)
    input_data = np.expand_dims(input_data, axis=0)
    print(input_data.shape)
    print("===========================")
    i = i + 1
    print(i)

    if floating_model:
      input_data = (np.float32(input_data) - args.input_mean) * args.input_std
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_data.astype('float32').tofile(newname)
    stop = time.time()
    print("time(s):", stop-start)