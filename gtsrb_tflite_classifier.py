#!pip install tensorflow_hub

from PIL import Image
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#import zipfile
#import tensorflow_hub as hub
from tensorflow.keras import layers
import pandas as pd 
from sklearn.metrics import accuracy_score
import tflite_runtime.interpreter as tflite
import platform
import time
import argparse

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]


def make_interpreter(model_file):
  model_file, *device = model_file.split('@')
  print("Model file = ",model_file)
  print("Device = ",device)
  return tflite.Interpreter(
      model_path=model_file,
      experimental_delegates=[
          tflite.load_delegate(EDGETPU_SHARED_LIB,
                               {'device': device[0]} if device else {})
      ])


print(tf.__version__)

IMAGE_SHAPE = (224, 224)

TESTING_DATA_DIR = "/home/casu/EESAM/datasets/GTSRB/JPG/testing_jpg"

tmp_test_labels_csv = "/home/casu/EESAM/datasets/GTSRB/JPG/GT-final_test.csv"
test_data_frame = pd.read_csv(tmp_test_labels_csv, header=0, sep=';')
test_data_frame['Filename'] = test_data_frame['Filename'].str.replace('.ppm','.jpg')
test_data_frame['ClassId'] = test_data_frame['ClassId'].astype(str).str.zfill(5)

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255,validation_split=0.1)




label_map = {
    0: '20_speed',
    1: '30_speed',
    2: '50_speed',
    3: '60_speed',
    4: '70_speed',
    5: '80_speed',
    6: '80_lifted',
    7: '100_speed',
    8: '120_speed',
    9: 'no_overtaking_general',
    10: 'no_overtaking_trucks',
    11: 'right_of_way_crossing',
    12: 'right_of_way_general',
    13: 'give_way',
    14: 'stop',
    15: 'no_way_general',
    16: 'no_way_trucks',
    17: 'no_way_one_way',
    18: 'attention_general',
    19: 'attention_left_turn',
    20: 'attention_right_turn',
    21: 'attention_curvy',
    22: 'attention_bumpers',
    23: 'attention_slippery',
    24: 'attention_bottleneck',
    25: 'attention_construction',
    26: 'attention_traffic_light',
    27: 'attention_pedestrian',
    28: 'attention_children',
    29: 'attention_bikes',
    30: 'attention_snowflake',
    31: 'attention_deer',
    32: 'lifted_general',
    33: 'turn_right',
    34: 'turn_left',
    35: 'turn_straight',
    36: 'turn_straight_right',
    37: 'turn_straight_left',
    38: 'turn_right_down',
    39: 'turn_left_down',
    40: 'turn_circle',
    41: 'lifted_no_overtaking_general',
    42: 'lifted_no_overtaking_trucks'
}



MODEL_FILE = "retrained_graph_mv1_100_224_ptq_edgetpu.tflite"


parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
      '-m', '--model', required=True, help='File path of .tflite file.')
parser.add_argument(
      '-b', '--batches', default=100, required=False, type=int, help='Number of image batches (32 each).')

args = parser.parse_args()

interpreter = make_interpreter(args.model)
interpreter.allocate_tensors()



input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
scale, zero_point = input_details[0]['quantization']




image_test_data = image_generator.flow_from_dataframe(test_data_frame, x_col="Filename",
                                                        directory=TESTING_DATA_DIR, y_col="ClassId",
                                                        target_size=IMAGE_SHAPE)
score = 0.0
count = 0
tot_time = 0.0
NUM_BATCHES = args.batches
print("\nStarting inference on {} batches...".format(NUM_BATCHES), end='', flush=True)
for image_test_batch, label_test_batch in image_test_data:
  #print("Image batch shape: ", image_test_batch.shape)
  #print("Label batch shape: ", label_test_batch.shape)

  batch_size = image_test_batch.shape[0]
  predicted_id = np.zeros(batch_size)
  label_id = np.argmax(label_test_batch, axis=-1)


  for i, image in enumerate(np.split(image_test_batch, batch_size)):
    #interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.set_tensor(input_details[0]['index'], np.uint8(image / scale + zero_point))

    start = time.perf_counter()
    interpreter.invoke()
    inference_time = time.perf_counter() - start
    tot_time = tot_time + inference_time
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_id[i] = np.argmax(output_data)

  #print("Accuracy of the shown eval batch, with the TensorFlow Lite model:")
  score += accuracy_score(label_id, predicted_id)
  count += 1
  if count == NUM_BATCHES:
    break
  #print(score)
  #print(predicted_id)
  #print(label_id)

score = score / count
tot_time = 1000.0 * tot_time / (count * batch_size)
print("Done!\n")
#print("Batches = ",count)
print("Accuracy = %.3f" % float(score))
print("Time per inference = %.2f ms" % float(tot_time))
quit()

