import cv2
import numpy as np
import math
from math import cos, sin, radians
import sympy
from djitellopy import Tello
from set_height import go_to_height
import time
import os


def line_equation(point1, point2):
    """ Calculate coefficients A, B, and C for the line equation Ax + By = C. """
    A = point2[1] - point1[1]
    B = point1[0] - point2[0]
    C = A * point1[0] + B * point1[1]
    return A, B, C

def find_intersection(l1, l2):
    """ Find the intersection point of two lines if it exists. """
    # Get the coefficients of the first line
    A1, B1, C1 = line_equation(l1[0], l1[1])

    # Get the coefficients of the second line
    A2, B2, C2 = line_equation(l2[0], l2[1])

    # Calculate the determinant
    determinant = A1 * B2 - A2 * B1

    if determinant == 0:
        return None  # Lines are parallel, no intersection

    # Calculate the x and y coordinates
    x = (C1 * B2 - C2 * B1) / determinant
    y = (A1 * C2 - A2 * C1) / determinant

    return (int(x), int(y))

def distance_btw_points(pt1, pt2):
    return np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)

def handend_point(intersection, pt1, pt2):
    d1 = distance_btw_points(intersection, pt1)
    d2 = distance_btw_points(intersection, pt2)
    if(d1 > d2):
        return pt1
    else:
        return pt2

def calculate_angle(pt1, pt2, pt3):

    # Create vectors
    vector_a = (pt1[0] - pt2[0], pt1[1] - pt2[1])
    vector_b = (pt3[0] - pt2[0], pt3[1] - pt2[1])

    # Calculate dot product
    dot_product = vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1]

    # Calculate the magnitudes of the vectors
    magnitude_a = math.sqrt(vector_a[0]**2 + vector_a[1]**2)
    magnitude_b = math.sqrt(vector_b[0]**2 + vector_b[1]**2)

    # Calculate the cosine of the angle
    cos_angle = dot_product / (magnitude_a * magnitude_b)

    # Calculate the angle in radians and then convert to degrees
    angle = math.acos(cos_angle)
    angle_degrees = math.degrees(angle)

    return angle_degrees

def point_position(pt1, pt2, pt3):

    # Create vectors
    vector_a = (pt2[0] - pt1[0], pt2[1] - pt1[1])  # Vector from pt1 to pt2
    vector_b = (pt3[0] - pt1[0], pt3[1] - pt1[1])  # Vector from pt1 to pt3

    # Calculate the cross product
    cross_product = vector_a[0] * vector_b[1] - vector_a[1] * vector_b[0]

    # Determine the position
    if cross_product > 0:
        return -1 #left of line
    elif cross_product < 0:
        return 1 #right of line
    else:
        return 0 # on the line
    
def draw(img, corners, imgpts):
    # corner = tuple(corners[1].ravel())
    corner_int = [int(x) for x in corners]
    corner = tuple(corner_int)
    # print(imgpts[1].ravel())
    for i in range(0,3):
        x,y = imgpts[i].ravel()
        if ( abs(x) > 1000 or abs(y)> 1000 ):
            imgpts[i] = corner[1]
    # print(imgpts)
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

def processimagedyn(corners, output_image, centeroid):
    corners = np.squeeze(corners)
    corners[:2] = corners[:2][np.argsort(corners[:2, 0])]
    corners[2:4] = corners[2:4][np.argsort(corners[2:4, 0])]
    center_coordinates = np.zeros((4,2),dtype=np.float32)
    print("corners", corners)
    for i in range(0,4):
        center_coordinates[i] = corners[i]
    if (len(corners)>=4):
        
        # Camera intrinsic parameters
        camera_matrix = np.array([[903.16, 0, 470.08], [0,903.47,352.76], [0, 0, 1]], dtype=np.float32)
        # camera_matrix = np.array([[453.7565, 0, 236.287], [0,454.004,176.50], [0, 0, 1]], dtype=np.float32)

        # Distortion coefficients
        # dist_coeffs = np.array([0.0080735,-0.15635, 0.0016456, -0.0003128, 0.04376], dtype=np.float32)
        dist_coeffs = np.array([0.02727,-0.241922, 0.00222, 0.000633, 0.5754], dtype=np.float32)
        # Your corresponding 2D image coordinates of the cube
        # object_points = np.array([[0,650,0], [0,0, 0], [650,650, 0], [650,0,0]], dtype=np.float32)
        object_points = np.array([[-420,420,0], [420,420, 0], [-420,-420, 0], [420,-420,0]], dtype=np.float32)
        # print(center_coordinates)
        # Solve for the pose

        success, rotation_vector, translation_vector = cv2.solvePnP(object_points, center_coordinates, camera_matrix, dist_coeffs)
        axis = np.float32([[100,0,0], [0,100,0], [0,0,100]]).reshape(-1,3)
        imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector , camera_matrix, dist_coeffs)
        float_array = np.array(imgpts)
        # Convert the float array to an integer array
        imgpts = float_array.astype(int)
        
        # Display the image with circles and sorted contours
        # print(tuple(center_coordinates[0].ravel()))
        o_img = draw(output_image,centeroid,imgpts)
    return o_img,rotation_vector,translation_vector


def dynamicframe_imageprocess(frame):
    # frame = cv2.imread('/home/pear/AerialRobotics/Aerial/HW5/dynamic/drone_images2/frame_226.png')
    # frame = frame[100:800,:]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range of blue color in HSV
    #                         50
    lower_blue = np.array([70,65,51])
    upper_blue = np.array([148,255,255]) #V: 255 -> 100

    lower_pink = np.array([120, 90, 0])
    upper_pink = np.array([179, 255, 255])

    # Threshold the HSV image to get only blue colors
    bluemask_binary = cv2.inRange(hsv, lower_blue, upper_blue)
    bluecolor_segment = cv2.bitwise_and(frame,frame, mask=bluemask_binary)
    image_color_b = cv2.cvtColor(bluemask_binary, cv2.COLOR_GRAY2BGR)

    # Threshold the HSV image to get only pink colors
    pinkmask_binary = cv2.inRange(hsv, lower_pink, upper_pink)
    pinkcolor_segment = cv2.bitwise_and(frame,frame, mask=pinkmask_binary)
    image_color_p = cv2.cvtColor(pinkmask_binary, cv2.COLOR_GRAY2BGR)

    # Combine both masks
    combined_mask = cv2.bitwise_or(bluemask_binary, pinkmask_binary)
    image_color = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)

    contours, _ = cv2.findContours(bluemask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # # Assuming the largest external contour corresponds to the square window border, get its corners
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.1 * cv2.arcLength(largest_contour, True)
    approx_corners_unsorted = cv2.approxPolyDP(largest_contour, epsilon, True)
    approx_corners = approx_corners_unsorted[approx_corners_unsorted[:, 0, 1].argsort()]

    # rotated_rect = cv2.minAreaRect(largest_contour)

    # Get the corner points of the rectangle
    # box = cv2.boxPoints(rotated_rect)
    # box = np.int0(box)

    # Draw the contours and the min area rectangle on the original image
    # cv2.drawContours(image_color_b, [largest_contour], -1, (0, 255, 0), 2)
    # cv2.drawContours(image_color_b, [approx_corners_unsorted], -1, (0, 0, 255), 2)
    # for corner in box:
    #     cv2.circle(image_color_b, tuple(corner), 5, (255, 0, 0), -1)
    # print(approx_corners)
    # # corner_image = image_color_b.copy()

    # approx_corners = appx_best_fit_ngon(bluemask_binary)
    # print(approx_corners)
    for corner in approx_corners:
        # print(corner)
        cv2.circle(frame, corner[0], 5, (0,255, 0), -1)

    # if not ret:
    #     break
    
    # Calculate the centroid of the window by averaging the x and y coordinates of the corners
    centroid = np.mean(approx_corners.reshape(-1, 2), axis=0)
    centroid = tuple(np.round(centroid).astype(int))
    upper_center = np.mean(approx_corners[:2], axis=0)
    upper_center = tuple(np.round(upper_center[0]).astype(int))
    lower_center = np.mean(approx_corners[2:4], axis=0)
    lower_center = tuple(np.round(lower_center[0]).astype(int))

    # output_image,rotation_vector,translation_vector = processimagedyn(approx_corners, frame, centroid )
    # print('tranlation', translation_vector)
    # cv2.imshow('mask', output_image)
    # cv2.waitKey(0)
    # Create a copy of the image to draw the vertical line
    # image_with_vertical_line = corner_image.copy()



    # pink image processing

    contours, _ = cv2.findContours(pinkmask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    # Draw the minimum area rectangle around the largest contour
    min_area_rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(min_area_rect)
    box = np.int0(box)
    # image_color = image_color_p.copy()
    cv2.drawContours(image_color, [box], 0, (0, 255, 0), 2)
    center = min_area_rect[0]
    width, height = min_area_rect[1]
    angle = min_area_rect[2]

    if width < height:
        angle += 90

    # Calculate the points for the center line
    pt1 = (int(center[0] + cos(radians(angle)) * max(width, height) / 2),
        int(center[1] + sin(radians(angle)) * max(width, height) / 2))
    pt2 = (int(center[0] - cos(radians(angle)) * max(width, height) / 2),
        int(center[1] - sin(radians(angle)) * max(width, height) / 2))

    # Draw the center line on the image


    l1 = (lower_center, upper_center)  # First line endpoints
    l2 = (pt1, pt2)  # Second line endpoints
    intersection = find_intersection(l1, l2)
    hand_end_pt = handend_point(intersection, pt1, pt2)
    # image_color = frame.copy()
    cv2.line(frame, intersection, upper_center, (0, 255, 255), 5)
    cv2.line(frame, intersection, hand_end_pt, (0, 0,255), 2)

    angle = calculate_angle(upper_center, intersection, hand_end_pt)
    position_flag = point_position(upper_center, intersection, hand_end_pt)
    if position_flag > 0:
        angle = 360 - angle
    text = f"Angle from vertical: {angle:.2f} degrees"
    cv2.putText(frame, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    # Display the frame
    # cv2.imshow('Video Frame', blue_segment)
    # cv2.imshow('mask', frame)
    # cv2.waitKey(0)
    return angle, frame,approx_corners, centroid
# Break the loop if 'q' is pressed

def run_dynamicwindow(tello,obj):

    output_directory = '/home/pear/AerialRobotics/Aerial/HW5/allwindows/src/outputs/dynamicwindow/processed_frames'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)

    go_to_height(tello, 150)
    goal = 0
    count = 0
    while goal == 0:
        try:
            # Get a frame from the Tello video stream
            for i in range(0,3):
                drone_frame = obj.frame
            
            if drone_frame is not None:
                drone_frame = cv2.cvtColor(drone_frame, cv2.COLOR_BGR2RGB)
                angle, frame,approx_corners, centroid = dynamicframe_imageprocess(drone_frame)
                processed_image_path = os.path.join(output_directory, f'processed_{count}.png')
                cv2.imwrite(processed_image_path, frame)
                print(f'saved_frame_{count}')
                count +=1
                print (angle)
                # Save the frame to the output directory with a timestamp
                if (angle > 150 and angle < 230 ):

                    output_image,rotation_vector,translation_vector = processimagedyn(approx_corners, frame, centroid )
                    
                    if translation_vector.mean() != 0:
                        print(tello.get_height())
                        print(int(translation_vector[0]*0.1),(int(translation_vector[2]*0.1))+30,(int(translation_vector[1]*0.1)))
                        # tello.go_xyz_speed((int(translation_vector[2]*0.1))+ 20,-(int(translation_vector[0]*0.1)), -int(translation_vector[0]*0.1), 40)
                        tello.go_xyz_speed((int(translation_vector[2]*0.1))+ 30,-(int(translation_vector[0]*0.1)), -50,80)
                        goal = 1
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

    return 3
