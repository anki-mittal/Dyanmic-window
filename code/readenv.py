import re
import numpy as np
import transformations as tf
from scipy.spatial.transform import Rotation as R

class Boundary:
    def __init__(self, xmin, ymin, zmin, xmax, ymax, zmax):
        self.xmin = xmin
        self.ymin = ymin
        self.zmin = zmin
        self.xmax = xmax
        self.ymax = ymax
        self.zmax = zmax

class Window:
    def __init__(self, parameters):
        # parameters should be a list of the numbers defining the window
        (self.x, self.y, self.z, self.xdelta, self.ydelta, self.zdelta, 
         self.qw, self.qx, self.qy, self.qz, self.xangdelta, self.yangdelta, 
         self.zangdelta) = parameters
        self.array = [self.x, self.y, self.z, self.qw, self.qx, self.qy, self.qz]

def read_environment_file(filepath):
    boundaries = []
    windows = []

    with open(filepath, 'r') as file:
        for line in file:
            # Skip comments and empty lines
            if line.startswith('#') or not line.strip():
                continue

            # Using regex to capture the numbers
            numbers = list(map(float, re.findall(r'-?\d+\.?\d*', line)))

            if 'boundary' in line:
                if len(numbers) == 6:
                    boundaries.append(Boundary(*numbers))
                else:
                    print("Error: incorrect number of elements for boundary definition.")
            elif 'window' in line:
                if len(numbers) == 13:
                    windows.append(Window(numbers))
                else:
                    print("Error: incorrect number of elements for window definition.")
    
    return boundaries, windows

def get_windows(file_path):
    # Filepath for the environment file

    x_pos_shift =[]
    boundaries, windows = read_environment_file(file_path)

    # Print out the parsed information
    # for boundary in boundaries:
    #     # print(f"Boundary: {vars(boundary)}")

    windowss = np.zeros((3,6))
    count = 0
    for window in windows:
        windowss[count,0:3] = window.array[0:3]
        quat = window.array[3:7]
        quat = quat / np.linalg.norm(quat)
        rotation = R.from_quat(quat)
        euler_angles = rotation.as_euler('xyz', degrees=True)
        # print(euler_angles)
        windowss[count,3:6] = euler_angles
        # quat
        count +=1

    
    x_pos_shift.append(int((0 - windowss[0,0])*100))
    x_pos_shift.append(int((windowss[0,0] - windowss[1,0])*100))
    x_pos_shift.append(int((windowss[1,0] - windowss[2,0])*100))
    print("x_shift",x_pos_shift)
    return x_pos_shift

if __name__ == "__main__":
    file_path = '/home/pear/AerialRobotics/Aerial/Unet_sim2real/src/worldmap/environment.txt'
    get_windows(file_path)