#first step: find pixels which correspond to the pen (purple?) 

import cv2 
import numpy as np 
import pyrealsense2 as rs 



def main(): 


    try:
        pipe = rs.pipeline() 
        config = rs.config() 

        pipeline_wrapper = rs.pipeline_wrapper(pipe)

        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        config.enable_stream(rs.stream.color,640,480, rs.format.bgr8,30)

        pipe.start(config)
        while True: 
        # for i in range(1): 
            frames = pipe.wait_for_frames()
            color_frame = frames.get_color_frame() 
            color_img = np.asanyarray(color_frame.get_data())

            hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)


            purple_lower = np.array([110,80,10,np.uint8])
            purple_upper = np.array([125,220,160],np.uint8)

            hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, purple_lower, purple_upper)
            result = cv2.bitwise_and(color_img, color_img, mask=mask)

            cv2.imwrite("test.jpg",color_img)
            cv2.imshow("img",result)
            cv2.waitKey(1) 
    finally:
        pipe.stop()
        cv2.destroyAllWindows() # close all windows



   

main() 