from tflite_runtime.interpreter import Interpreter
from PIL import Image
import tensorflow as tf
import numpy as np
import time

def load_labels(path): # Read the labels from the text file as a Python list.
  with open(path, 'r') as f:
    return [line.strip() for i, line in enumerate(f.readlines())]

def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image
  input_tensor.shape, input_tensor

def classify_image(interpreter, image, top_k=1):
  set_input_tensor(interpreter, image)

  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  print("output details: ", output_details)
  output = np.squeeze(interpreter.get_tensor(output_details['index']))

  scale, zero_point = output_details['quantization']
  output = scale * (output - zero_point)
  print("output after processing \n", output.shape, output)

  ordered = np.argpartition(-output, 1)
  return [(i, output[i]) for i in ordered[:top_k]][0]

model_path = "model/transfer_model_tflite_2023-02-24_15-14.tflite"
label_path = "model/trash_classes.txt"

interpreter = Interpreter(model_path=model_path)
print("Model loaded succesfully!")

interpreter.allocate_tensors()
_, height, width, _ = interpreter.get_input_details()[0]['shape']
print("Input shape: ", height, width)

# Load sample image to be classified
image = Image.open("img/plastic193.jpg").convert('RGB').resize((width, height))

print(image)

# using tflite to predict image class
time_start = time.time()
label_id, prob = classify_image(interpreter, image)
time_end = time.time()
print("Prediction time: ", np.round(time_end - time_start, 3))
print("Label ID: ", label_id)
# Load labels
labels = load_labels(label_path)
cl_label = labels[label_id]
print(f"Predicted class: {cl_label} with Accuracy: {np.round(prob*100, 2)} %")