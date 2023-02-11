import cv2
from gpiozero import Button
from datetime import datetime
from time import sleep

button = Button(17)
video_capture = cv2.VideoCapture(0)
# Check success
if not video_capture.isOpened():
    raise Exception("Could not open video device")
button.wait_for_press()
# Read picture. ret === True on success
sleep(1)
ret, frame = video_capture.read()
cv2.imwrite(f'./img/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.jpg', frame)
# Close device
video_capture.release()
