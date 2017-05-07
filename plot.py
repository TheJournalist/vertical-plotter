import cv2
import numpy as np
import pygame
from pygame.locals import *
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
gpio.setcfg(port.PA0, gpio.OUTPUT)   # SERVO

# gpio
gpio.output(port.PA11, gpio.LOW)
gpio.output(port.PA12, gpio.LOW)
gpio.output(port.PA1, gpio.LOW)
gpio.output(port.PA6, gpio.LOW)
gpio.output(port.PA0, gpio.LOW)

x = 5.05
dx = 0.1
y = 8.63
dy = 0.0
D = 40.1
L = 10.0
R = 36.0
dL = 0.0
dR = 0.0
somaPixels = 0

def display(str):
    text = font.render(str, True, (255, 255, 255), (159, 182, 205))
    textRect = text.get_rect()
    textRect.centerx = screen.get_rect().centerx
    textRect.centery = screen.get_rect().centery

    screen.blit(text, textRect)
    pygame.display.update()


def moveTeste():
    global x
    global dx
    global y
    global dy
    global D
    global L
    global R
    global dL
    global dR

    while x < 20.0:
        dL = (np.sqrt(L*L + 2*x*dx + 2*y*dy + dx*dx + dy*dy) - L) 
        if dL > 0:
            moveL(int(np.rint(dL/ 0.003575)), gpio.LOW)        
        else:
            moveL(int(np.rint(-dL/ 0.003575)), gpio.HIGH)
        L = L + dL
        print("dL - " + str(dL/ 0.003575)+ " " + str(np.rint(dL/ 0.003575))) 
        
        dR = (np.sqrt(R*R + 2*(x-D)*dx + 2*y*dy + dx*dx + dy*dy) - R) 
        if dR > 0:
            moveR(int(np.rint(dR/ 0.00365)), gpio.HIGH)        
        else:
            moveR(int(np.rint(-dR/ 0.00365)), gpio.LOW)
        R = R + dR
        print("dR - " + str(dR/ 0.00365) + " " + str(np.rint(dR/ 0.00365)))

        x = x + dx
        y = y + dy
        print("x - " + str(x))

    return

def move(steps, dir1, dir2):
    # direction
    gpio.output(port.PA1, dir1)
    gpio.output(port.PA11, dir2)
    # steps
    for i in range(steps):
        gpio.output(port.PA12, gpio.HIGH)
        gpio.output(port.PA6, gpio.HIGH)
        sleep(0.0001)
        gpio.output(port.PA12, gpio.LOW)
        gpio.output(port.PA6, gpio.LOW)
        sleep(0.0001)
    return

def moveL(steps, dir1):
    # direction
    gpio.output(port.PA1, dir1)
    # steps
    for i in range(steps):
        gpio.output(port.PA6, gpio.HIGH)
        sleep(0.0001)
        gpio.output(port.PA6, gpio.LOW)
        sleep(0.0001)
    return

def moveR(steps, dir1):
    # direction
    gpio.output(port.PA11, dir1)
    # steps
    for i in range(steps):
                
        gpio.output(port.PA12, gpio.HIGH)
        sleep(0.0001)
        gpio.output(port.PA12, gpio.LOW)
        sleep(0.0001)
    return

def moveLeft(steps):
    move(steps, gpio.HIGH, gpio.HIGH)
    return

def moveRight(steps):
    move(steps, gpio.LOW, gpio.LOW)
    return

def moveUp(steps):
    move(steps, gpio.HIGH, gpio.LOW)
    return

def moveDown(steps):
    move(steps, gpio.LOW, gpio.HIGH)
    
    return

def servo(toggled):
    if toggled:
        gpio.output(port.PA0, gpio.HIGH)
    else:
        gpio.output(port.PA0, gpio.LOW)
    return



pygame.init()
screen = pygame.display.set_mode((640,480))
pygame.display.set_caption('Plotter!')
screen.fill((159, 182, 205))
font = pygame.font.Font(None, 17)
num = 0
done = False


moveTeste()

#esquerda
#moveR(2500, gpio.LOW)
#moveL(242, gpio.HIGH)

#direita
#moveL(242, gpio.LOW)
#moveR(2500, gpio.HIGH)

#moveDown(2000)

while not done:
    display(str(num))
    num += 1

    pygame.event.pump()
    keys = pygame.key.get_pressed()
    if keys[K_ESCAPE]:
        done = True
    if keys[K_d]:
        moveRight(1)
    if keys[K_a]:
        moveLeft(1)
    if keys[K_w]:
        moveUp(1)
    if keys[K_s]:
        moveDown(1)

''' #---------------------------------------------------------------------------------
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
image = cv2.resize(image, (160,120))


#height, width = image.shape
#for i in range(width/4):
#    for j in range(height/2):
      

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
'''

