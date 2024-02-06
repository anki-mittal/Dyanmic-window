from djitellopy import Tello
import time



def go_to_height(tello, target_height):
    threshold = 11  # Allow for a threshold of 5 cm
    current_height = tello.get_height()
    desired_height = target_height - current_height

    if desired_height > threshold:
        print(f"Going up to {target_height} cm.")
        # tello.move_up(desired_height)
        tello.go_xyz_speed(0,0,desired_height,80)
    elif desired_height < -threshold:
        print(f"Going down to {target_height} cm.")
        # tello.move_down(-desired_height)
        tello.go_xyz_speed(0,0,-desired_height,80)

    time.sleep(2)  # Adjust this sleep time based on your drone's response time

    while True:
        current_height = tello.get_height()
        height_difference = target_height - current_height

        # Check if the current height is within the threshold range
        if -threshold <= height_difference <= threshold:
            print(f"Reached target height: {current_height} cm.")
            break
        # If not within the threshold, adjust the height accordingly
        elif height_difference > threshold:
            print(f"Adjusting height, still too low by {height_difference} cm.")
            # tello.move_up(height_difference)
            tello.go_xyz_speed(0,0,desired_height,80)
        elif height_difference < -threshold:
            print(f"Adjusting height, still too high by {-height_difference} cm.")
            # tello.move_down(-height_difference)
            tello.go_xyz_speed(0,0,-desired_height,80)
        # Wait before checking the height again
        time.sleep(2)  # Adjust this based on your drone's response time
    print(f"Drone is at the Correct Height !!!")

if __name__ == "__main__":
    tello = Tello()
    tello.connect() 
    tello.takeoff()
    go_to_height(tello, 180)
    print(tello.get_height())
    time.sleep(2)
    tello.land()