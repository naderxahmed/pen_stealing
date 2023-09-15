import cv2
import numpy as np
import pyrealsense2 as rs

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
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=50)
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

def main():
    try:
        # Initialize RealSense pipeline
        pipe = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        profile = pipe.start(config)

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
                if cv2.arcLength(cnt, True) > 100:
                    cv2.drawContours(color_img, contours[0:2], -1, (0, 255, 0), 1)

                    if len(contours) == 2 and are_contours_near(contours[0], contours[1], max_distance=30, max_size_diff=3):
                        top_center, bottom_center = compute_center(contours[0]), compute_center(contours[1])
                        average_x = int((top_center[0] + bottom_center[0]) / 2)
                        average_y = int((top_center[1] + bottom_center[1]) / 2)
                        center = (average_x, average_y)

                    elif len(contours) == 1:
                        center = compute_center(contours[0])

                    if center:
                        found_pen = True
                        tracked_center = center
                        print("distance", aligned_depth_frame.get_distance(center[0], center[1]))
                    
                        # Display location and distance
                        cv2.putText(color_img, "Location: {}, {}".format(center[0], center[1]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        depth = aligned_depth_frame.get_distance(center[0], center[1])
                        cv2.putText(color_img, "Distance: {:.2f} meters".format(depth), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                        cv2.circle(color_img, center, 10, (0, 0, 255), -1)


            # If we didn't find the pen in this frame, use the tracked center from the previous frame
            if not found_pen and tracked_center:
                center = tracked_center

            cv2.imshow("img", color_img)
            cv2.waitKey(1)

    finally:
        pipe.stop()
        cv2.destroyAllWindows()  # Close all windows

if __name__ == "__main__":
    main()
