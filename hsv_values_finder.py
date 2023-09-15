import cv2
import numpy as np

def track_hsv_color(image_path):
    def placeholder_callback(x):
        pass

    # Load image
    image = cv2.imread(image_path)

    # Create a window
    cv2.namedWindow('Image')

    # Create trackbars for HSV color range adjustments
    cv2.createTrackbar('Hue Min', 'Image', 0, 179, placeholder_callback)
    cv2.createTrackbar('Saturation Min', 'Image', 0, 255, placeholder_callback)
    cv2.createTrackbar('Value Min', 'Image', 0, 255, placeholder_callback)
    cv2.createTrackbar('Hue Max', 'Image', 0, 179, placeholder_callback)
    cv2.createTrackbar('Saturation Max', 'Image', 0, 255, placeholder_callback)
    cv2.createTrackbar('Value Max', 'Image', 0, 255, placeholder_callback)

    # Set default values for max HSV trackbars
    cv2.setTrackbarPos('Hue Max', 'Image', 179)
    cv2.setTrackbarPos('Saturation Max', 'Image', 255)
    cv2.setTrackbarPos('Value Max', 'Image', 255)

    while True:
        hue_min = cv2.getTrackbarPos('Hue Min', 'Image')
        saturation_min = cv2.getTrackbarPos('Saturation Min', 'Image')
        value_min = cv2.getTrackbarPos('Value Min', 'Image')
        hue_max = cv2.getTrackbarPos('Hue Max', 'Image')
        saturation_max = cv2.getTrackbarPos('Saturation Max', 'Image')
        value_max = cv2.getTrackbarPos('Value Max', 'Image')

        # Set minimum and maximum HSV values
        lower_hsv = np.array([hue_min, saturation_min, value_min])
        upper_hsv = np.array([hue_max, saturation_max, value_max])

        # Convert to HSV format and apply color thresholding
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
        filtered_image = cv2.bitwise_and(image, image, mask=mask)

        # Display filtered image
        cv2.imshow('Image', filtered_image)

        # Press 'q' to exit the loop
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# Usage:
if __name__ == "__main__":
    image_path = 'test.jpg'
    track_hsv_color(image_path)
