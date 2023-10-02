import cv2

# Load the pre-trained detection models
car_cascade = cv2.CascadeClassifier(
    "C:/Users/ADMIN/Desktop/face/cardetect/car_detector.xml"
)
pedestrian_cascade = cv2.CascadeClassifier("haarcascade_fullbody.xml")
bike_cascade = cv2.CascadeClassifier(
    "C:/Users/ADMIN/Desktop/face/cardetect/two_wheeler.xml"
)
bus_cascade = cv2.CascadeClassifier(
    "C:/Users/ADMIN/Desktop/face/cardetect/Bus_front.xml"
)

# Open the video file for detection
video1 = cv2.VideoCapture(
    "C:/Users/ADMIN/Desktop/face/cardetect/15519-264715970 (360p).mp4"
)

while True:
    # Read the current frame from the video
    successful_frame_read, frame = video1.read()

    if not successful_frame_read:
        break

    # Convert the frame to grayscale for detection
    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect and label cars
    car_coordinates = car_cascade.detectMultiScale(grayscaled_frame)
    for x, y, w, h in car_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 245), 2)
        cv2.putText(
            frame, "Car", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
        )

    # Detect and label pedestrians
    pedestrian_coordinates = pedestrian_cascade.detectMultiScale(grayscaled_frame)
    for x, y, w, h in pedestrian_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            "Pedestrian",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    # Detect and label bikes
    bike_coordinates = bike_cascade.detectMultiScale(grayscaled_frame)
    for x, y, w, h in bike_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(
            frame, "Bike", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
        )

    # Detect and label buses
    bus_coordinates = bus_cascade.detectMultiScale(grayscaled_frame)
    for x, y, w, h in bus_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame, "Bus", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

    # Display the frame with detected objects
    cv2.imshow("Object Detection", frame)

    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(3) & 0xFF == ord("q"):
        break

# Release the video capture and close the window
video1.release()
cv2.destroyAllWindows()
