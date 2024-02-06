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


# Paths
global grayscale_output_folder,rgb_output_folder,output_directory,env_path,continous_frames

grayscale_output_folder = '/home/pear/AerialRobotics/Aerial/HW5/allwindows/src/outputs/checker_window/grayscale_output_folder'
if not os.path.exists(grayscale_output_folder):
    os.makedirs(grayscale_output_folder)

rgb_output_folder = '/home/pear/AerialRobotics/Aerial/HW5/allwindows/src/outputs/checker_window/rgb_output_folder'
if not os.path.exists(rgb_output_folder):
    os.makedirs(rgb_output_folder)

output_directory = '/home/pear/AerialRobotics/Aerial/HW5/allwindows/src/outputs/checker_window/output_directory'
if not os.path.exists(output_directory):
    os.makedirs(output_directory, exist_ok=True)

continous_frames = '/home/pear/AerialRobotics/Aerial/HW5/allwindows/src/outputs/checker_window/continous_frames'
if not os.path.exists(continous_frames):
    os.makedirs(continous_frames, exist_ok=True)

env_path = '/home/pear/AerialRobotics/Aerial/Unet_sim2real/src/worldmap/environment.txt'


def capture_continuous_photos(output_directory, frame_read_obj, stop_event):
    count = 0
    while not stop_event.is_set():
        frame = frame_read_obj.frame
        if frame is not None:
            filename = os.path.join(continous_frames, f'photo_{count}.jpg')
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename, frame)
            count += 1

def get_center(drone_frame,count,model, checkpoint, net, device):
    count +=1
    
    bgr_frame = cv2.cvtColor(drone_frame, cv2.COLOR_RGB2BGR)
    frame_filename = os.path.join(output_directory, f'frame_{count}.png')
    cv2.imwrite(frame_filename, bgr_frame)
    print(f'saved rgb_frame_{count}')
    
    #corner prediction

    drone_frame = cv2.resize(drone_frame, (480,360))
    pil_image = Image.fromarray(drone_frame)
    # plt.imshow(pil_image)
    # plt.pause(1)

    pred = get_prediction_image(pil_image, model, checkpoint, net, device)
    print("got predicted images")
    grayscale_image = np.where(pred == 1, 255, 0).astype(np.uint8)
    processed_image_path = os.path.join(grayscale_output_folder, f'processed_{count}.png')
    cv2.imwrite(processed_image_path, grayscale_image)
    print(f'saved predicted_grayscale_frame_{count}')

    # Process the grayscale image along with the original image
    corners_add_count = 0
    drone_frame = cv2.cvtColor(drone_frame, cv2.COLOR_BGR2RGB)
    output_image,rotation_vector, translation_vector = processimage(grayscale_image,drone_frame,corners_add_count = 0)
    processed_rgb_image_path = os.path.join(rgb_output_folder, f'processed_rgb_{count}.png')
    cv2.imwrite(processed_rgb_image_path, output_image)
    rotation_vector = np.squeeze(rotation_vector)
    rotation = R.from_rotvec(rotation_vector)
    euler_angles = rotation.as_euler('zyx', degrees= True)  # 'zyx' is the sequence of rotation; change as needed
    print(f'saved gate orientaion frame_{count}')
    print("euler angles=", euler_angles)
    print("translation vector",translation_vector)

    return translation_vector, rotation_vector,count


def run_checkerwindows(tello,obj, model, checkpoint, net, device):
    desired_altitude = 150

    x_pos_shift = [90, -210]
    z_pos_shift = [80, 50]
    f_pos_shift = [30,-40]
    # x_pos_shift = get_windows(env_path)

    rotate = [1,50]                    ## CHANGE HERE FOR FOR GO TO HEIGHT

    ## uncomment for recoding frames
    # stop_photo_thread = threading.Event()
    # photo_thread = Thread(target=capture_continuous_photos, args=(continous_frames, obj, stop_photo_thread))
    # photo_thread.start()
    window = [False, False]
    count = 0
    for w_count in range(0,2):
        # tello.go_xyz_speed(f_pos_shift[w_count],x_pos_shift[w_count],z_pos_shift[w_count],90)
        # go_to_height(tello, desired_altitude)
        if w_count !=0: ## CHANGE HERE FOR FOR GO TO HEIGHT
            tello.rotate_clockwise(rotate[w_count])
        time.sleep(1.5)
        while window[w_count] == False:
            try:
                # Get a frame from the Tello video stream
                for i in range(0,3):
                    drone_frame = obj.frame
                
                if drone_frame is not None:
                    # Save the frame to the output directory with a timestamp
                    translation_vector, rotation_vector,count = get_center(drone_frame,count,model, checkpoint, net, device)
                    
                    # print(translation_vector.mean())
                    if translation_vector.mean() != 0:
                        print(tello.get_height())
                        print(int(translation_vector[0]*0.1),(int(translation_vector[2]*0.1))+30,(int(translation_vector[1]*0.1)))
                        # tello.go_xyz_speed((int(translation_vector[2]*0.1))+ 20,-(int(translation_vector[0]*0.1)), -int(translation_vector[0]*0.1), 40)
                        tello.go_xyz_speed((int(translation_vector[2]*0.1))+ 25,-(int(translation_vector[0]*0.1)), -30,90) # time changed to 100 from 80 
                        window[w_count] = True
                    # tello.rotate_clockwise(rotate[w_count])
                    time.sleep(0.5)

            except KeyboardInterrupt:
                # HANDLE KEYBOARD INTERRUPT AND STOP THE DRONE COMMANDS
                print('keyboard interrupt')
                # stop_photo_thread.set()
                # photo_thread.join()
                tello.streamoff= 0
                tello.emergency()
                tello.emergency()
                tello.end()
                tello.land()
                break

    return 1



    
