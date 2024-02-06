######################
# IMPORTING MODULES
######################
import sys
import threading
from djitellopy import Tello
import time
import cv2
import os
from flask import Flask, send_file
import os
from pathlib import Path
sys.path.append('/home/pear/AerialRobotics/Aerial/HW5/allwindows/src/pytorch-spynet')
import run
import logging
from image_processing import process_image, process_image_two

# PATHS


# Directory where raw images from drone camera are saved
images_directory_1 = '/home/pear/AerialRobotics/Aerial/HW5/allwindows/src/outputs/opticalwall/frames_raw_images'
# Directory where optical flow images during navigation are saved
images_directory_2 = '/home/pear/AerialRobotics/Aerial/HW5/allwindows/src/outputs/opticalwall/frames_opticalflow'


# THREAD 1 : Take Images from 2 folders and publish them on a Flask Server

# Flask for creating a server
app = Flask(__name__)
logging.getLogger('werkzeug').setLevel(logging.ERROR)

# Indexing Last Image from a Folder
def get_latest_image_set1():
    list_of_files = Path(images_directory_1).glob('*.jpg')  # Now looking for .png files
    latest_file = max(list_of_files, key=os.path.getctime, default=None)
    return latest_file
def get_latest_image_set2():
    list_of_files = Path(images_directory_2).glob('*.png')  # Now looking for .jpg files
    latest_file = max(list_of_files, key=os.path.getctime, default=None)
    return latest_file

# Retreiving Images from the Folder
@app.route('/latest-image-set1')
def latest_image_set1():
    latest_image_path = get_latest_image_set1()
    if latest_image_path:
        return send_file(str(latest_image_path), mimetype='image/jpg')  # Updated MIME type
    else:
        return "No images found", 404
@app.route('/latest-image-set2')
def latest_image_set2():
    latest_image_path = get_latest_image_set2()
    if latest_image_path:
        return send_file(str(latest_image_path), mimetype='image/png')  # Updated MIME type
    else:
        return "No images found", 404

# Starting a Thread where these images are published on a server
def run_flask_app():
    app.run(host='0.0.0.0', port=8000, use_reloader=False, debug=False)
# flask_thread = threading.Thread(target=run_flask_app)
# flask_thread.start()


def visual_servoing(tello, cx, cy, image_center_x=480, image_center_y=360, threshold=80, position_scale=1, speed = 100, forward_distance = 250):
    # Calculating the difference between target and image center. 
    delta_x = image_center_x - cx
    delta_y = image_center_y - cy

    # Applying threshold 
    if abs(delta_x) < threshold:
        delta_x = 0
    if abs(delta_y) < threshold:
        delta_y = 0

    # Calculating the position Adjustment.  
    # position_x = int(delta_x * position_scale)
    # position_y = int(delta_y * position_scale)
    if (delta_x != 0):
        # if(abs(delta_x)<300):
        #     position_x = int((int(delta_x)/abs(delta_x))*30)
        # else:
        #     position_x = int((int(delta_x)/abs(delta_x))*60)
        # position_x = int((int(delta_x)/abs(delta_x))*((abs(delta_x)+70)/6))
        position_x = int((int(delta_x)/abs(delta_x))*((abs(delta_x)+40)/6)) 
    else:
        position_x = 0
    if (delta_y != 0):
        # if(abs(delta_y)<300):
        #     position_y = int((int(delta_y)/abs(delta_y))*30)  
        # else:
        #     position_y = int((int(delta_x)/abs(delta_y))*60)
        # position_y = int((int(delta_y)/abs(delta_y))*((abs(delta_y)+70)/6))
        position_y = int((int(delta_y)/abs(delta_y))*((abs(delta_y)+40)/6))
    else:
        position_y = 0

    # Move drone based on the delta values. 
    if (delta_x != 0) or (delta_y != 0):
        print("aligning the center")
        tello.go_xyz_speed(0, (position_x), (position_y), speed) # cx, cy are are in the image axis, not drone axis
        return True
    # If already aligned, move the robot straight through the hole. 
    else:
        print("Going fordward")  
        tello.go_xyz_speed(forward_distance, 0,0,100)
        return False



def run_opticalwall(tello, obj):
    time.sleep(1)
    count = 1
    img_count = 1
    print("drone height:",tello.get_height())
    images = [None, None, None]
    task_status = True
    while task_status:
        try:
            # Get a frame from the Tello video stream
            fone = False
            ftwo = False
            for i in range(0,5):
                drone_frame = obj.frame
            while fone == False:
                drone_frame = obj.frame
                if drone_frame is not None:
                    frameone = drone_frame
                    fone = True
            # image_one_path = os.path.join(drone_img_path, f'imageone{count}.png')
            frameone = cv2.cvtColor(frameone, cv2.COLOR_RGB2BGR)
            print("got frame1")
            # cv2.imwrite(image_one_path, frameone)
            time.sleep(1)

            while ftwo == False:
                drone_frame = obj.frame
                if drone_frame is not None:
                    frametwo = drone_frame
                    ftwo = True
            # image_two_path = os.path.join(drone_img_path, f'imagetwo{count}.png')
            frametwo = cv2.cvtColor(frametwo, cv2.COLOR_RGB2BGR)
            print("got frame2")
            # cv2.imwrite(image_two_path, frametwo)
            img = run.get_opticalflow(frameone, frametwo)
            output_image_path = os.path.join(images_directory_2, f'flow{img_count}.png')
            img_count = img_count + 1
            cv2.imwrite(output_image_path, img)
            print(f"Image saved as {output_image_path}")

            images[count-1] = img
            if(count%3 == 0):
                binary_image, cx, cy = process_image(images)
                # binary_image, cx, cy = process_image_two(images)
                count = count + 1
                output_image_path = os.path.join(images_directory_2, f'flow{img_count}.png')
                cv2.imwrite(output_image_path, binary_image)
                print(f"Image saved as {output_image_path}")
                # tello.move_up(20)
                # tello.move_down(20)
                images = [None, None, None]
                count = 0
                task_status = visual_servoing(tello, cx, cy, image_center_x=480, image_center_y=200, threshold=80, position_scale=1, speed = 80, forward_distance =240)
                time.sleep(2)
            count = count +1
            img_count +=1
            

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

    return 2