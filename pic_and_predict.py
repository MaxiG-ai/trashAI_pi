import numpy as np
import time
import cv2
from tflite_runtime.interpreter import Interpreter
from PIL import Image
from gpiozero import Button
from datetime import datetime
from time import sleep

# Taking a picture, once the button is pressed
button = Button(17)
video_capture = cv2.VideoCapture(0)
# Check success
if not video_capture.isOpened():
    raise Exception("Could not open video device")
button.wait_for_press()
# Read picture. ret === True on success
sleep(1)
ret, frame = video_capture.read()
pic_name = f'./img/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.jpg'
cv2.imwrite('pic_name', frame)
# Close device
video_capture.release()


# Read the labels from the text file as a Python list.
def load_labels(path): 
  with open(path, 'r') as f:
    return [line.strip() for i, line in enumerate(f.readlines())]

# Set the input tensor.
def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image

# Run inference and get the output tensor.
def classify_image(interpreter, image, top_k=1):
  set_input_tensor(interpreter, image)

  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))

  scale, zero_point = output_details['quantization']
  output = scale * (output - zero_point)

  ordered = np.argpartition(-output, 1)
  return [(i, output[i]) for i in ordered[:top_k]][0]

# defining input paths for tflite model and labels
model_path = "test_model_mobile_net/mobilenet_v1_1.0_224_quant.tflite"
label_path = "test_model_mobile_net/labels_mobilenet_quant_v1_224.txt"

# Load the TFLite model and allocate tensors.   
interpreter = Interpreter(model_path=model_path)
print("Model loaded succesfully!")

interpreter.allocate_tensors()
_, height, width, _ = interpreter.get_input_details()[0]['shape']
print("Input shape: ", height, width)

# Load sample image to be classified
image = Image.open("img/2023-02-11_16-04-39.jpg").convert('RGB').resize((width, height))

# using tflite to predict image class
time_start = time.time()
label_id, prob = classify_image(interpreter, image)
time_end = time.time()
print("Prediction time: ", np.round(time_end - time_start, 3))

# Load labels
labels = load_labels(label_path)
cl_label = labels[label_id]
print(f"Predicted class: {cl_label} with Accuracy: {np.round(prob*100, 2)} %")

