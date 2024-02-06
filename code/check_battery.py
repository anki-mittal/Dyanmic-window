from djitellopy import Tello
from dynamicwindow import run_dynamicwindow
import time
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

tello = Tello()
tello.connect()
battery_level = tello.get_battery()
print(f"Battery level: {battery_level}%")
tello.streamon()
obj = tello.get_frame_read(with_queue= True)
intialisation(obj)
tello.takeoff()
time.sleep(3)
tello.go_xyz_speed(30,100,80,90)
tello.go_xyz_speed(255,1,-30,90)
tello.go_xyz_speed(-40,-220,50,90)
tello.go_xyz_speed(224,12,-30,100)
tello.go_xyz_speed(10,150,-50,100)
tello.rotate_counter_clockwise(30)
tello.go_xyz_speed(0,0,-20,90)
tello.go_xyz_speed(240,0,0,100)
tello.go_xyz_speed(0,0,70,80)
flag3 = run_dynamicwindow(tello,obj)
tello.land()