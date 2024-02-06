import cv2
import numpy as np

def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def predict_forth_corner(image, filtered_contours,corners_add_count):
    avg_area = np.mean([cv2.contourArea(contour) for contour in filtered_contours])
    patch_size = int(np.sqrt(avg_area))
    # patch_size = 50
    # Extract centroids for each of the filtered contours
    corners_add_count +=1
    centers = []
    for contour in filtered_contours:
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centers.append((cX, cY))

    # Calculate pairwise distances between corners
    distances = {}
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            dist = calculate_distance(centers[i], centers[j])
            distances[(i,j)] = dist

    # Sort distances
    sorted_distances = sorted(distances.items(), key=lambda x: x[1])

    # Identify the two shortest distances. These correspond to the two sides of the window that meet at the missing corner.
    side1 = sorted_distances[0]
    side2 = sorted_distances[1]

    # Check which two corners are common in the identified sides. This will give the corner from which the sides originate.
    common_corners = list(set(side1[0]).intersection(side2[0]))
    if len(common_corners) != 1:
        raise ValueError("Could not uniquely determine the origin corner for the missing side")

    origin_corner_index = common_corners[0]
    other_corners = list(set(side1[0] + side2[0]) - {origin_corner_index})

    # Calculate the vectors representing the two known sides
    vector1 = (centers[other_corners[0]][0] - centers[origin_corner_index][0], centers[other_corners[0]][1] - centers[origin_corner_index][1])
    vector2 = (centers[other_corners[1]][0] - centers[origin_corner_index][0], centers[other_corners[1]][1] - centers[origin_corner_index][1])

    # Calculate the vector for the missing side using vector addition
    missing_vector = (vector1[0] + vector2[0], vector1[1] + vector2[1])

    # Estimate the fourth corner using the missing vector
    fourth_corner = (centers[origin_corner_index][0] + missing_vector[0], centers[origin_corner_index][1] + missing_vector[1])

    # print("Fourth Corner:", fourth_corner)

    # Add a 5x5 white patch at the fourth corner
    # patch_size = 30
    half_patch = patch_size // 2
    image[int(fourth_corner[1]-half_patch):int(fourth_corner[1]+half_patch), int(fourth_corner[0]-half_patch):int(fourth_corner[0]+half_patch)] = 255

    return image,corners_add_count


def draw(img, corners, imgpts):
    # corner = tuple(corners[1].ravel())
    corner_int = [int(x) for x in corners]
    corner = tuple(corner_int)
    # print(imgpts[1].ravel())
    for i in range(0,3):
        x,y = imgpts[i].ravel()
        if ( abs(x) > 500 or abs(y)> 500 ):
            imgpts[i] = corner[1]
    print(imgpts)
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

def processimage(thresh,output_image,corners_add_count):
    # Find contours of the white patches
    rotation_vector = np.zeros((1,3))
    translation_vector = np.zeros((1,3))
    if(corners_add_count>=2):
        return output_image,rotation_vector,translation_vector
    kernel = np.ones((2,2),np.uint8)  # adjust kernel size if needed
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    # cv2.imshow('Result', dilated)
    # cv2.waitKey(0)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 100
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_area]
    # print(len(contours))
    
    if (len(filtered_contours)>=4):
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]
        contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[:2])
        contours_left = contours[:2]
        contours_left = sorted(contours_left, key=lambda c: cv2.boundingRect(c)[1])
        contours_right = contours[2:4]
        contours_right = sorted(contours_right, key=lambda c: cv2.boundingRect(c)[1])
        # Create an empty canvas for drawing
        
        center_coordinates = np.zeros((4,2),dtype=np.float32)
        corners = []
        # Calculate and print the center coordinates for each contour and draw them
        count = 0
        for contour in contours_left:
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                # print(f"Center Coordinates: ({cx}, {cy})")
                corners.append((cx,cy))
                center_coordinates[count] = np.array([cx,cy])
                count = count+1
                # Draw a small circle at the center point
                cv2.circle(output_image, (cx, cy), 5, (0, 0, 255), -1)  # -1 means fill the circle
        
        for contour in contours_right:
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                # print(f"Center Coordinates: ({cx}, {cy})")
                corners.append((cx,cy))
                center_coordinates[count] = np.array([cx,cy])
                count = count+1
                # Draw a small circle at the center point
                cv2.circle(output_image, (cx, cy), 5, (0, 0, 255), -1)  # -1 means fill the circle

        diag1 = (center_coordinates[0] + center_coordinates[3])/2
        diag2 = (center_coordinates[1] + center_coordinates[2])/2
        window_center = (diag2+diag1)/2

        # Camera intrinsic parameters
        # camera_matrix = np.array([[903.16, 0, 235.04], [0,903.47,176.38], [0, 0, 1]], dtype=np.float32)
        camera_matrix = np.array([[453.7565, 0, 236.287], [0,454.004,176.50], [0, 0, 1]], dtype=np.float32)

        # Distortion coefficients
        # dist_coeffs = np.array([0.0080735,-0.15635, 0.0016456, -0.0003128, 0.04376], dtype=np.float32)
        dist_coeffs = np.array([0.02727,-0.241922, 0.00222, 0.000633, 0.5754], dtype=np.float32)
        # Your corresponding 2D image coordinates of the cube
        # object_points = np.array([[0,650,0], [0,0, 0], [650,650, 0], [650,0,0]], dtype=np.float32)
        object_points = np.array([[-325,325,0], [-325,-325, 0], [325,325, 0], [325,-325,0]], dtype=np.float32)
        # print(center_coordinates)
        # Solve for the pose
        success, rotation_vector, translation_vector = cv2.solvePnP(object_points, center_coordinates, camera_matrix, dist_coeffs)
        axis = np.float32([[400,0,0], [0,400,0], [0,0,400]]).reshape(-1,3)
        imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector , camera_matrix, dist_coeffs)
        float_array = np.array(imgpts)
        # Convert the float array to an integer array
        imgpts = float_array.astype(int)
        
        # Display the image with circles and sorted contours
        # print(tuple(center_coordinates[0].ravel()))
        o_img = draw(output_image,window_center,imgpts)
        return o_img,rotation_vector,translation_vector
    if (len(filtered_contours)==3):
        fourcorner_binary_image,corner_add_count= predict_forth_corner(thresh, filtered_contours,corners_add_count) 
        output_image,rotation_vector,translation_vector = processimage(fourcorner_binary_image,output_image,corner_add_count)
    return output_image,rotation_vector,translation_vector
        

if __name__ == "__main__":
    image = cv2.imread('/home/pear/AerialRobotics/Aerial/Unet_sim2real/grayscale_output/processed_6.png')
    output_image = cv2.imread('/home/pear/AerialRobotics/Aerial/Unet_sim2real/videoframes/trial3/frame_6.png')
    output_image = cv2.resize(output_image,(480,360))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    corners_add_count = 0
    o_img,rotation_vector,translation_vector = processimage(thresh,output_image,corners_add_count)
    cv2.imshow('Result', o_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()