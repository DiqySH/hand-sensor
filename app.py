from utils import get_average_knuckle_distance, get_root_distance, get_palm_coordinate, map_range
from detect import detect_hands

import cv2
import numpy as np
import socket
import threading
import sys
import time

# Change set this higher if a closed hand gets detected as open
MIN_HAND_OPEN_DISTANCE = 0.078

# Protection value for when the AI detects 1 hand as 2:
MULTI_HAND_PROTECTION = 0.15

# TCP server settings
TCP_IP = 'localhost'
TCP_PORT = 2314
TCP_TICKRATE = 30
BUFFER_SIZE = 8192

# Inner detection margins
MARGIN_HORIZONTAL = 70
MARGIN_VERTICAL = 70

# For the coordinate mapper
MAPPED_WIDTH = 600
MAPPED_HEIGHT = 400

mouse_is_down = False
start_point = (0, 0)
margined_start_point = (0, 0)
end_point = (0, 0)
margined_end_point = (0, 0)

# Statuses to send to the client
hands_detected = 0
hand_open = False
palm_coordinates = (0, 0)

# Flag for the server to determine whether or not to send packets
send_payload = False


def mouse_event(event, x, y, flags, param):
    global start_point, end_point, mouse_is_down, margined_start_point, margined_end_point
    if event == cv2.EVENT_LBUTTONDOWN:
        start_point = (x, y)
        end_point = (x, y)
        mouse_is_down = True
    elif event == cv2.EVENT_MOUSEMOVE and mouse_is_down:
        end_point = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_is_down = False
        if start_point[0] > end_point[0] or start_point[1] > end_point[1]:
            start_point, end_point = end_point, start_point
        if end_point[0] - start_point[0] > MARGIN_HORIZONTAL and end_point[1] - start_point[1] > MARGIN_VERTICAL:
            margined_start_point = (start_point[0] + MARGIN_HORIZONTAL, start_point[1] + MARGIN_VERTICAL)
            margined_end_point = (end_point[0] - MARGIN_HORIZONTAL, end_point[1] - MARGIN_VERTICAL)
        else:
            margined_start_point = start_point
            margined_end_point = end_point


def server_handling():
    # While loop to keep tring to re-establish a connection
    while(True):
        # Wait for incoming client
        client, ip = server.accept()
        print("Client connected")
        while(True):
            try:
                # Construct the payload and send it. Wait a tick length before sending the next packet
                payload = f"X{palm_coordinates[0]},Y{palm_coordinates[1]},H{hands_detected},O{1 if hand_open else 0}\n"
                client.send(payload.encode())
                time.sleep(1 / TCP_TICKRATE)
            except:
                # Presumably the payload failed to send, so the client has lost connection
                print("Client disconnected")
                break
        # Close connection with the client upon disconnect
        client.close()


# Set up webcam capturing and mouse events
cap = cv2.VideoCapture(0)
cv2.namedWindow("Hand detection")
cv2.setMouseCallback("Hand detection", mouse_event)

# Set up TCP server
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((TCP_IP, TCP_PORT))
server.listen()
server_thread = threading.Thread(target=server_handling)
server_thread.setDaemon(True)
server_thread.start()

while cap.isOpened():
    success, image = cap.read()
    if not success:
        # If reading the frame failed, continue with the loop and don't send a status update
        continue

    # Flip the image horizontally for a later selfie-view display
    image = cv2.flip(image, 1)

    # Rotate the image clockwise
    # image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    # Process this frame with the model
    success, multi_hand_landmarks, image = detect_hands(
        image, start_point, end_point)

    # Draw the detection region
    detection_region = np.zeros(image.shape, np.uint8)
    cv2.rectangle(detection_region, start_point,
                  end_point, (0, 0, 255), cv2.FILLED)
    image = cv2.addWeighted(image, 1, detection_region, 0.10, 1)
    if not mouse_is_down:
        margin_region = np.zeros(image.shape, np.uint8)
        cv2.rectangle(margin_region, margined_start_point,
            margined_end_point, (0, 255, 255), cv2.FILLED)
        image = cv2.addWeighted(image, 1, margin_region, 0.20, 1)


    # Draw the hand annotations on the image
    if success and multi_hand_landmarks:
        if len(multi_hand_landmarks) > 1 and get_root_distance(multi_hand_landmarks[0].landmark[0], multi_hand_landmarks[1].landmark[0]) > MULTI_HAND_PROTECTION:
            # If we've detected more than 1 hand, update the multiple hands status and draw an alert on screen
            cv2.putText(image, "Multiple hands detected! " + str(len(multi_hand_landmarks)), (20, 50),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
            hands_detected = len(multi_hand_landmarks)
        else:
            # Cool, we detected a single hand. Update the coordinates and open status
            landmarks = multi_hand_landmarks[0].landmark

            # Get the hand open status
            if get_average_knuckle_distance(landmarks) >= MIN_HAND_OPEN_DISTANCE:
                hand_open = True
            else:
                hand_open = False
            palm_coordinates_unmapped = get_palm_coordinate(landmarks)
            cv2.putText(image,
                        "Hand open" if hand_open else "Hand closed", (20, 70), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

            # Get the coordinates of the palm
            draw_palm_coordinates = (int(palm_coordinates_unmapped[0] * image.shape[1]), int(palm_coordinates_unmapped[1] * image.shape[0]))

            mapped_palm_x = (int)(map_range(
                palm_coordinates_unmapped[0], margined_start_point[0] / image.shape[1], margined_end_point[0] / image.shape[1], 0, MAPPED_WIDTH))
            mapped_palm_y = (int)(map_range(
                palm_coordinates_unmapped[1], margined_start_point[1] / image.shape[0], margined_end_point[1] / image.shape[0], 0, MAPPED_HEIGHT))
            palm_coordinates = (mapped_palm_x, mapped_palm_y)

            # Draw a purple circle at the palm
            cv2.circle(image, draw_palm_coordinates,
                       15, (255, 0, 255), cv2.FILLED)

            hands_detected = 1
    else:
        hands_detected = 0
    cv2.imshow('Hand detection', image)
    send_payload = True
    if cv2.waitKey(5) & 0xFF == 27:
        break
cv2.destroyAllWindows()
cap.release()
server.close()
sys.exit()  # This will also kill the server thread
