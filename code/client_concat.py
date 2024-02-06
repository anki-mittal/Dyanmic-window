# import cv2
# import numpy as np
# import requests
# import time

# # import zmq
# # URLs for the images (replace with your actual URLs)
# url1 = 'http://130.215.217.143:8000/latest-image-set1'
# url2 = 'http://130.215.217.143:8000/latest-image-set2'

# def combine_and_display_images(url1,url2):
#     cv2.namedWindow("Combined Image", cv2.WINDOW_NORMAL)  # Create a window

#     while True:
#         try:
#             response1 = requests.get(url1)
#             response2 = requests.get(url2)

#             if response1.status_code == 200 and response2.status_code == 200:
#                 image1 = cv2.imdecode(np.frombuffer(response1.content, np.uint8), cv2.IMREAD_COLOR)
#                 image2 = cv2.imdecode(np.frombuffer(response2.content, np.uint8), cv2.IMREAD_COLOR)

#                 if image1 is not None and image2 is not None and image1.shape == image2.shape and image1.dtype == image2.dtype:
                
#                     combined_image = cv2.hconcat([image1, image2])  # Horizontal concatenation
#                     cv2.imshow("Combined Image", combined_image)
#                     # cv2.imshow("Combined Image", image1)
#                 else:
#                     print("Invalid or incompatible images")

#                 if cv2.waitKey(1) & 0xFF == ord('q'):  # press 'q' to quit the loop
#                     break
#             else:
#                 print("Failed to retrieve images from one or both URLs")

#             time.sleep(0.5)  # Wait for 0.05 seconds
#         except:
#             print("Exception Raised")
#     cv2.destroyAllWindows()  # Destroy the window outside the loop

# combine_and_display_images(url1, url2)


import cv2
import numpy as np
import requests
import time
import os

# URLs for the images (replace with your actual URLs)
url1 = 'http://130.215.217.143:8000/latest-image-set1'
url2 = 'http://130.215.217.143:8000/latest-image-set2'

def combine_and_display_images(url1, url2):
    cv2.namedWindow("Combined Image", cv2.WINDOW_NORMAL)  # Create a window

    # Directory to save images
    save_directory = "/home/pear/AerialRobotics/Aerial/HW4/pytorch-spynet/images/frames_concatenated"
    os.makedirs(save_directory, exist_ok=True)  # Create directory if it doesn't exist

    image_counter = 0  # To name the saved images uniquely

    while True:
        try:
            response1 = requests.get(url1)
            response2 = requests.get(url2)

            if response1.status_code == 200 and response2.status_code == 200:
                image1 = cv2.imdecode(np.frombuffer(response1.content, np.uint8), cv2.IMREAD_COLOR)
                image2 = cv2.imdecode(np.frombuffer(response2.content, np.uint8), cv2.IMREAD_COLOR)

                if image1 is not None and image2 is not None and image1.shape == image2.shape and image1.dtype == image2.dtype:
                    combined_image = cv2.hconcat([image1, image2])  # Horizontal concatenation
                    cv2.imshow("Combined Image", combined_image)
                    # cv2.imshow("Combined Image", image1)

                    # Save the combined image
                    # save_path = os.path.join(save_directory, f"combined_image_{image_counter}.jpg")
                    # cv2.imwrite(save_path, combined_image)
                    # image_counter += 1

                else:
                    print("Invalid or incompatible images")

                if cv2.waitKey(1) & 0xFF == ord('q'):  # press 'q' to quit the loop
                    break
            else:
                print("Failed to retrieve images from one or both URLs")

            time.sleep(0.001)  # Wait for 0.2 seconds
        except Exception as e:
            print(f"Exception Raised: {e}")
    cv2.destroyAllWindows()  # Destroy the window outside the loop

combine_and_display_images(url1,url2)