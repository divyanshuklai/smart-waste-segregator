import cv2

# Initialize the camera (0 is usually the default camera)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Camera not accessible")
    exit()

# Capture a frame
ret, frame = cap.read()

# Save the captured frame if successful
if ret:
    cv2.imwrite('opencv_test_image.jpg', frame)
    print("Image captured as opencv_test_image.jpg")
else:
    print("Error: Could not read frame")

# Release the camera and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()

