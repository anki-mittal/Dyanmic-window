from flask import Flask, send_file
import os
from pathlib import Path
import logging 

app = Flask(__name__)
logging.getLogger('werkzeug').setLevel(logging.ERROR)
# Directory where raw images from drone camera are saved
images_directory_1 = '/home/pear/AerialRobotics/Aerial/HW4/pytorch-spynet/images/frames_raw_images'
# Directory where optical flow images during navigation are saved
images_directory_2 = '/home/pear/AerialRobotics/Aerial/HW4/pytorch-spynet/images/frames_opticalflow'

#############################
# LAST IMAGE FROM THE FOLDER
#############################
def get_latest_image_set1():
    list_of_files = Path(images_directory_1).glob('*.jpg')  # Now looking for .png files
    latest_file = max(list_of_files, key=os.path.getctime, default=None)
    return latest_file
def get_latest_image_set2():
    list_of_files = Path(images_directory_2).glob('*.png')  # Now looking for .jpg files
    latest_file = max(list_of_files, key=os.path.getctime, default=None)
    return latest_file

########################
# RETRIEVING IMAGES
########################
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

if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=8000)
    app.run(host='0.0.0.0', port=8000, use_reloader=False, debug=True)
