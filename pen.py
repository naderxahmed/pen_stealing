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
        config.enable_stream(rs.stream.depth,640,480, rs.format.z16,30)


        profile = pipe.start(config)

        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: " , depth_scale)

        # align_to = rs.stream.color
        # align = rs.align(align_to)

        while True: 
            frames = pipe.wait_for_frames()
            color_frame = frames.get_color_frame() 
            color_img = np.asanyarray(color_frame.get_data())

            hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)

            purple_lower = np.array([110,80,10],np.uint8)
            purple_upper = np.array([125,220,160],np.uint8)

            hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, purple_lower, purple_upper)
            result = cv2.bitwise_and(color_img, color_img, mask=mask)

            kernel = np.ones((5, 5), np.uint8)
            result = cv2.erode(result, kernel, iterations=2)
            result = cv2.dilate(result,kernel,iterations=4)

            temp = cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)
            contours, hierarchy = cv2.findContours(temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            

            contours = sorted(contours,key=cv2.contourArea,reverse=True)

            
            cv2.drawContours(result, contours[0:2], -1, (0,255,0), 1)

            if len(contours) == 2: 

                top_center, bottom_center = compute_center(contours[0]), compute_center(contours[1])
                average_x = int((top_center[0]+bottom_center[0])/2)
                average_y = int((top_center[1]+bottom_center[1])/2)
                cv2.circle(result, (average_x,average_y), 5, (0,0,255),2)

            elif len(contours)==1: 
                center = compute_center(contours[0]) 
                cv2.circle(result, center, 5, (0,0,255),2)


            # for cnt in contours:
            #     print(len(contours))
            #     print(cv2.contourArea(cnt))
                # if len(cnt) >=5 and cv2.arcLength(cnt,True) >=200: 
                    # ellipse = cv2.fitEllipse(cnt)
                    # cv2.ellipse(result, ellipse, (255,0, 255), 1, cv2.LINE_AA)
 
                    
         
            cv2.imshow("img",result)
            cv2.waitKey(1) 
            
    finally:
        pipe.stop()
        cv2.destroyAllWindows() # close all windows



def compute_center(points): 
    M = cv2.moments(points)
    if M['m00'] != 0:
        return (int(M['m10']/M['m00']),int(M['m01']/M['m00'])) 
    

main() 