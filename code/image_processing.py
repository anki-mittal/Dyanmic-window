import cv2
from skimage.filters import threshold_otsu

def process_image(images):
    image1 = images[0]
    image2 = images[1]
    image3 = images[2]
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)
    print("1")
    threshold1 = threshold_otsu(image1)-10
    threshold2 = threshold_otsu(image2)-10
    threshold3 = threshold_otsu(image3)-10

    # Segment the image
    # Pixels darker than the threshold are set to 255 (white), others are 0 (black)
    binary1 = (image1 > threshold1).astype('uint8') * 255
    binary1 = cv2.bitwise_not(binary1)
    binary2 = (image2 > threshold2).astype('uint8') * 255
    binary2 = cv2.bitwise_not(binary2)
    binary3 = (image3 > threshold3).astype('uint8') * 255
    binary3 = cv2.bitwise_not(binary3)


    # cv2.imshow('binary', binary2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    common_black_regions = cv2.bitwise_and(binary1, binary2)
    common_black_regions = cv2.bitwise_and(common_black_regions, binary3)

    contours, _ = cv2.findContours(common_black_regions, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    height, width = common_black_regions.shape

    contours = [cnt for cnt in contours if not (
    cv2.pointPolygonTest(cnt, (0,0), False) >= 0 or
    cv2.pointPolygonTest(cnt, (0,height-1), False) >= 0 or
    cv2.pointPolygonTest(cnt, (width-1,0), False) >= 0 or
    cv2.pointPolygonTest(cnt, (width-1,height-1), False) >= 0 )]
    # Find the largest contour based on the area
    largest_contour = max(contours, key=cv2.contourArea)

    # Calculate the moments to find the centroid of the largest contour
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        # This means the contour area is zero and we can't divide by 0
        cX, cY = 0, 0

    # The coordinates cX, cY are the centroid of the largest white region
    print(f"The center of the largest white region is at: ({cX}, {cY})")

    # Optional: Draw the contour and the center on the image
    # Draw the largest contour
    output_image = cv2.cvtColor(common_black_regions, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(output_image, [largest_contour], -1, (0, 255, 0), 2)

    # Draw the centroid
    cv2.circle(output_image, (cX, cY), 7, (0, 0, 255), -1)
    cv2.circle(output_image, (480, 200), 7, (0, 255, 0), -1)
    cv2.circle(output_image, (480, 200), 80, (0, 255, 0), 2)
    return output_image, cX, cY
# Display the image with the drawn contour and centroid

def process_image_two(images):
    image1 = images[0]
    image2 = images[1]

    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)


    threshold1 = threshold_otsu(image1)-10
    threshold2 = threshold_otsu(image2)-10


    # Segment the image
    # Pixels darker than the threshold are set to 255 (white), others are 0 (black)
    binary1 = (image1 > threshold1).astype('uint8') * 255
    binary1 = cv2.bitwise_not(binary1)
    binary2 = (image2 > threshold2).astype('uint8') * 255
    binary2 = cv2.bitwise_not(binary2)
    # cv2.imshow('binary', binary2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    common_black_regions = cv2.bitwise_and(binary1, binary2)

    contours, _ = cv2.findContours(common_black_regions, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    height, width = common_black_regions.shape

    contours = [cnt for cnt in contours if not (
    cv2.pointPolygonTest(cnt, (0,0), False) >= 0 or
    cv2.pointPolygonTest(cnt, (0,height-1), False) >= 0 or
    cv2.pointPolygonTest(cnt, (width-1,0), False) >= 0 or
    cv2.pointPolygonTest(cnt, (width-1,height-1), False) >= 0 )]
    # Find the largest contour based on the area
    largest_contour = max(contours, key=cv2.contourArea)
    # Find the largest contour based on the area

    # Calculate the moments to find the centroid of the largest contour
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        # This means the contour area is zero and we can't divide by 0
        cX, cY = 0, 0

    # The coordinates cX, cY are the centroid of the largest white region
    print(f"The center of the largest white region is at: ({cX}, {cY})")

    # Optional: Draw the contour and the center on the image
    # Draw the largest contour
    output_image = cv2.cvtColor(common_black_regions, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(output_image, [largest_contour], -1, (0, 255, 0), 2)

    # Draw the centroid
    cv2.circle(output_image, (cX, cY), 7, (0, 0, 255), -1)
    cv2.circle(output_image, (480, 200), 7, (0, 255, 0), -1)
    cv2.circle(output_image, (480, 200), 80, (0, 255, 0), 2)
    return output_image, cX, cY



if __name__ == "__main__":
    image1 = cv2.imread('/home/pear/AerialRobotics/Aerial/HW4/pytorch-spynet/images/frames_opticalflow/flow1.png')
    image2 = cv2.imread('/home/pear/AerialRobotics/Aerial/HW4/pytorch-spynet/images/frames_opticalflow/flow2.png')
    image3 = cv2.imread('/home/pear/AerialRobotics/Aerial/HW4/pytorch-spynet/images/frames_opticalflow/flow3.png')
    images = [image1, image2, image3]
    # images = [image1, image2]
    output_image,cX, cY = process_image(images)
    cv2.imshow('Image with Largest Contour and Centroid', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
