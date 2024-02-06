import cv2
from djitellopy import Tello
import os
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from unet import MakeUnet
from unet import get_prediction_image, imshow_gray
import torch 
from PIL import Image 
import os
from imgprocessing import processimage
from scipy.spatial.transform import Rotation as R
from readenv import get_windows
from set_height import go_to_height
from multiprocessing import Process
from threading import Thread
import threading
from checker import run_checkerwindows
import sys
from pathlib import Path
sys.path.append('/home/pear/AerialRobotics/Aerial/HW5/allwindows/src/pytorch-spynet')
import run
import logging
from image_processing import process_image
from opticalwall import run_opticalwall
from dynamicwindow import run_dynamicwindow

def load_unetmodel():
    device = 'cuda:0'
    checkpoint = torch.load('/home/pear/AerialRobotics/Aerial/HW5/allwindows/src/Unetmodel/my_checkpoint_rutwik_training3.pth.tar')
    net, lossfun, optimizer = MakeUnet(False)
    print(f"Device Available is : {device}")
    model = net
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    return model, checkpoint, net, device

def intialisation(obj):
    img = obj.frame
    fone = False
    ftwo = False
    while img is None:
        img = obj.frame
    for i in range(0,20):
        img = obj.frame

    while fone == False:
        drone_frame = obj.frame
        if drone_frame is not None:
            frameone = drone_frame
            fone = True

    while ftwo == False:
        drone_frame = obj.frame
        if drone_frame is not None:
            frametwo = drone_frame
            ftwo = True

    img = run.get_opticalflow(frameone, frametwo)
    print("optical wall initalization complete")

if __name__ == "__main__":

    model, checkpoint, net, device = load_unetmodel()
    tello = Tello()
    tello.connect()
    tello.streamon()
    obj = tello.get_frame_read(with_queue= True)
    battery_level = tello.get_battery()
    print(f"Battery level: {battery_level}%")
    img = None
    img = obj.frame
    if img is not None:
        for i in range(0,6):
            img = obj.frame
            img = cv2.resize(img, (480,360))
        pil_image = Image.fromarray(img)
        pred = get_prediction_image(pil_image, model, checkpoint, net, device)
        print("unet model loaded")

    intialisation(obj)

    tello.takeoff()
    time.sleep(1)
    tello.go_xyz_speed(50,80,100,100)
    flag1 = run_checkerwindows(tello,obj,model, checkpoint, net, device)
    tello.go_xyz_speed(10,100,-40,100) # speed changed to 100 from 30
    tello.rotate_counter_clockwise(75)
    tello.go_xyz_speed(0,40,0,100)
    flag2 = run_opticalwall(tello,obj)
    tello.rotate_clockwise(85)
    # tello.go_xyz_speed(0,0,70,100)
    flag3 = run_dynamicwindow(tello,obj)
    tello.land()
    tello.end()

