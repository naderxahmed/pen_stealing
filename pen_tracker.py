import cv2
import numpy as np
import pyrealsense2 as rs
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
import modern_robotics as mr 
from multiprocessing import Process, Manager 

robot = InterbotixManipulatorXS("px100", "arm", "gripper")


def find_pen_color_mask(hsv_img):
    # Define the purple color range
    purple_lower = np.array([110, 80, 10], np.uint8)
    purple_upper = np.array([125, 220, 160], np.uint8)

    # Create a mask for the purple color
    return cv2.inRange(hsv_img, purple_lower, purple_upper)

def preprocess_image(image):
    # Erode, dilate, and close operations to clean the mask
    kernel = np.ones((3, 3), np.uint8)
    image = cv2.erode(image, kernel, iterations=2)
    image = cv2.dilate(image, kernel, iterations=6)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=5)
    return image

def find_pen_contours(image):
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return sorted(contours, key=cv2.contourArea, reverse=True)

def compute_center(points):
    # Compute the center of a set of points
    M = cv2.moments(points)
    if M['m00'] != 0:
        return (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
    

def are_contours_near(contour1, contour2, max_distance, max_size_diff):
    # Check if two contours are near each other and somewhat comparable in size
    center1 = compute_center(contour1)
    center2 = compute_center(contour2)
    distance = np.linalg.norm(np.array(center1) - np.array(center2))
    size_diff = cv2.contourArea(contour1) / cv2.contourArea(contour2)
    return distance < max_distance and size_diff < max_size_diff


#finds pen position in camera frame and displays its tracking
def pen_detection(d):
    try:
        # Initialize RealSense pipeline
        pipe = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        cfg = pipe.start(config)
        profile = cfg.get_stream(rs.stream.color)
        intr = profile.as_video_stream_profile().get_intrinsics()
    
        align_to = rs.stream.color
        align = rs.align(align_to)
        tracked_center = None
        center = None

        while True:
            frames = pipe.wait_for_frames()
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            depth_img = np.asanyarray(aligned_depth_frame.get_data())
            color_img = np.asanyarray(color_frame.get_data())

            # Convert the color image to HSV
            hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)

            # Find the purple color mask
            mask = find_pen_color_mask(hsv)

            # Preprocess the mask
            result = preprocess_image(mask)

            # Find contours in the mask
            contours = find_pen_contours(result)

            # Initialize a flag to indicate whether we found the pen
            found_pen = False

            for cnt in contours:
                if cv2.arcLength(cnt, True) > 250:
                    cv2.drawContours(color_img, contours[0:2], -1, (0, 255, 0), 1)

                    center = compute_center(contours[0])

                    if center:
                        found_pen = True
                        tracked_center = center
                        # print("distance", aligned_depth_frame.get_distance(center[0], center[1]))
                    
                        # Display location and distance
                        cv2.putText(color_img, "Location: {}, {}".format(center[0], center[1]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        depth = aligned_depth_frame.get_distance(center[0], center[1])
                        cv2.putText(color_img, "Distance: {:.2f} meters".format(depth), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                        cv2.circle(color_img, center, 10, (0, 0, 255), -1)

                        pen_camera_frame = rs.rs2_deproject_pixel_to_point(intr, [center[0], center[1]], depth)

                        d["pen_camera_frame"] = pen_camera_frame 
                        
            # If we didn't find the pen in this frame, use the tracked center from the previous frame
            if not found_pen and tracked_center:
                center = tracked_center

            cv2.imshow("img", color_img)
            cv2.waitKey(1)

    finally:
        pipe.stop()
        cv2.destroyAllWindows()  # Close all windows
        

#determine the deltas between camera and robot frames when robot arm is at known pen location 
#add deltas to camera frame to get to robot frame
def determine_deltas(pen_camera_frame): 

    joints = robot.arm.get_joint_commands()
    T = mr.FKinSpace(robot.arm.robot_des.M, robot.arm.robot_des.Slist, joints)
    [R, p] = mr.TransToRp(T) # get the rotation matrix and the displacement
    pen_robot_frame = [p[0],p[1],p[2]]
    print("pen_robot_frame",pen_robot_frame)
    
    prx, pry, prz = pen_robot_frame
    pcx, pcy, pcz = pen_camera_frame 
    soln = [prx-pcx, pry-pcz, prz+pcy]
    return soln



def main(): 

    with Manager() as manager: 
        d = manager.dict() 
        d["pen_camera_frame"] = None 

        #initiate pen detection thread, constantly updating d["pen_camera_frame"]
        p = Process(target=pen_detection,args=(d,))
        p.start() 
        
        mode = 'h'
        robot.gripper.release() 

        while mode != 'q':
            pen_camera_frame = d["pen_camera_frame"]

            mode = input("[h]ome, [s]leep, [q]uit, [c]alibrate, [p]en_tracking,") 
            if mode == 'h': 
                robot.arm.go_to_home_pose() 
            if mode == 's': 
                robot.arm.go_to_sleep_pose() 
            if mode == 'j': #print current robot position
                joints = robot.arm.get_joint_commands()
                T = mr.FKinSpace(robot.arm.robot_des.M, robot.arm.robot_des.Slist, joints)
                [R, p] = mr.TransToRp(T) # get the rotation matrix and the displacement
                print("current robot position",p)

            if mode == 'c': 
                if pen_camera_frame: 
                    robot.gripper.grasp() 
                    deltas = determine_deltas(pen_camera_frame) #determines deltas between camera and robot frame, assuming that the pen is currently placed in the gripper
                    robot.gripper.release() 

                else: 
                    print("Pen not found yet") 
                
            if mode =='p':
                goal_position = (pen_camera_frame[0]+deltas[0], pen_camera_frame[2]+deltas[1], -1*(-pen_camera_frame[2]+deltas[2]))
                print("goal position", goal_position)
                _, success = robot.arm.set_ee_pose_components(x=goal_position[0],y=goal_position[1],z=goal_position[2])
                if success: 
                    robot.gripper.grasp()  
                    robot.gripper.release()
                    print("current robot position",p)
                    print("GOAL POSITION",goal_position)
                else: 
                    joints = robot.arm.get_joint_commands()
                    T = mr.FKinSpace(robot.arm.robot_des.M, robot.arm.robot_des.Slist, joints)
                    [R, p] = mr.TransToRp(T) # get the rotation matrix and the displacement
                    print("current robot position",p)
                    print("GOAL POSITION",goal_position)

                    








if __name__ == "__main__":
    main()
