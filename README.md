# trashAI_pi
Local Files for trashAI Project on raspberry pi

## How to run
The file `takepicture.py` demonstrates the use of the camera in connection with the raspberry pi. Once the script is run, the Pi waits for the press of the button we installed on the GPIO. Then it will start a 5s timer and then take the picture. 

The file `tflite_tutorial.py` demonstrates the prediction of an image using only the tflite_runtime. This part is adapted from this [Blog post](https://blog.paperspace.com/tensorflow-lite-raspberry-pi/)

In the file `pic_and_predict.py` we implemented the two files above together and thus created the option to take a picture and directly predict it with the trained model. 

The models have to be loaded onto the raspberry pi via this GitHub-Repo. In the `img`-Folder we have some sample images for testing with the `tflite_tutorial.py`-Script.