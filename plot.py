import cv2
import numpy as np
from pyA20.gpio import gpio
from pyA20.gpio import port
from time import sleep

# initialize the gpio module
gpio.init()

# setup the ports
gpio.setcfg(port.PA11, gpio.OUTPUT)  # DIR1
gpio.setcfg(port.PA12, gpio.OUTPUT)  # STEP1
gpio.setcfg(port.PA1, gpio.OUTPUT)   # DIR2
gpio.setcfg(port.PA6, gpio.OUTPUT)   # STEP2

# gpio
gpio.output(port.PA11, gpio.LOW)
gpio.output(port.PA12, gpio.LOW)
gpio.output(port.PA1, gpio.LOW)
gpio.output(port.PA6, gpio.LOW)


dire = 0
sleep(0.5)
step = 0
time = 0.1

while True:
    # step
    gpio.output(port.PA12, gpio.HIGH)
    gpio.output(port.PA6, gpio.HIGH)
    sleep(0.0008)
    gpio.output(port.PA12, gpio.LOW)
    gpio.output(port.PA6, gpio.LOW)
    sleep(0.0008)
    step = step + 1
    
    # change direction
    if step > 2000:
        step = 0
        if dire:
            gpio.output(port.PA11, gpio.LOW)
            gpio.output(port.PA1, gpio.LOW)
            dire = 0
        else:
            gpio.output(port.PA11, gpio.HIGH)
            gpio.output(port.PA1, gpio.HIGH)
            dire = 1



# ---------------------------------------------------------------------------------
# image processing examples

# generating the kernels
kernel_sharpen_1 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
 
# Camera 0 is the integrated web cam on my netbook
camera_port = 0
 
#Number of frames to throw away while the camera adjusts to light levels
ramp_frames = 60
 
# Now we can initialize the camera capture object with the cv2.VideoCapture class.
# All it needs is the index to a camera port.
camera = cv2.VideoCapture(camera_port)

print("Taking image...")
# Captures a single image from the camera and returns it in PIL format
def get_image():
 # read is the easiest way to get a full image out of a VideoCapture object.
 retval, im = camera.read()
 return im
 
# Ramp the camera - these frames will be discarded and are only used to allow v4l2
# to adjust light levels, if necessary
for i in xrange(ramp_frames):
 temp = get_image()
# Take the actual image we want to keep
camera_capture = get_image()

print("Processing image...")
# grayscale
image = cv2.cvtColor(camera_capture, cv2.COLOR_BGR2GRAY)
# Pixelate
#image = cv2.resize(image, (100,75))

# applying different kernels to the input image
output_1 = cv2.filter2D(image, -1, kernel_sharpen_1)

cv2.imshow('Original', cv2.resize(image, (640,480)))
cv2.imshow('Sharpening', cv2.resize(output_1, (640,480)))
cv2.waitKey(0)

#file = "/home/orangepi/image.png"
# A nice feature of the imwrite method is that it will automatically choose the
# correct format based on the file extension you provide. Convenient!
#cv2.imwrite(file, camera_capture)
 
# You'll want to release the camera, otherwise you won't be able to create a new
# capture object until your script exits
del(camera)
