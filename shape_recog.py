import cv2

# Function to determine shape based on contour properties
def detect_shape(contour):
    # Calculate perimeter and area of contour
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)

    # Check if perimeter is zero
    if perimeter == 0:
        return "Unknown"

    # Calculate circularity (4 * pi * area / perimeter^2)
    circularity = (4 * 3.14 * area) / (perimeter * perimeter)

    # Check if circularity is within a certain threshold (for detecting circles)
    if circularity > 0.8:
        return "Circle"
    else:
        # Check number of vertices to determine shape
        vertices = len(cv2.approxPolyDP(contour, 0.03 * perimeter, True))
        if vertices == 3:
            return "Triangle"
        elif vertices == 4:
            # Check aspect ratio to differentiate between square and rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if 0.95 <= aspect_ratio <= 1.05:
                return "Square"
            else:
                return "Rectangle"
        elif vertices == 5:
            return "Pentagon"
        else:
            return "Unknown"

# Initialize camera
camera = cv2.VideoCapture(0)

# Check if camera opened successfully
if not camera.isOpened():
    print("Error: Failed to open camera.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = camera.read()

    # Check if frame is empty
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edged image
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours are found
    if contours:
        # Loop over contours
        for contour in contours:
            # Determine shape of contour
            shape = detect_shape(contour)

            # Check if shape is not "Unknown"
            if shape != "Unknown":
                # Draw contour and shape name on the frame
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                # Get bounding box coordinates for better text placement
                x, y, w, h = cv2.boundingRect(contour)
                cv2.putText(frame, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow('Shape Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera
camera.release()
cv2.destroyAllWindows()
