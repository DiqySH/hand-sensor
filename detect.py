
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
model = mp_hands.Hands(min_detection_confidence=0.4,
                       min_tracking_confidence=0.4)


def detect_hands(image, start_crop, end_crop):
    # Create a cropped detection image according to the detection region
    if start_crop[0] > end_crop[0] or start_crop[1] > end_crop[1]:
        start_crop, end_crop = end_crop, start_crop

    cropped_image = image[start_crop[1]:end_crop[1], start_crop[0]:end_crop[0]]

    # Convert the BGR image to RGB.
    if not (cropped_image.shape[0] is 0 or cropped_image.shape[1] is 0):
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        cropped_image.flags.writeable = False

        # Process the image with the model
        results = model.process(cropped_image)

        # Scales used for resizing the coordinates
        xScale = cropped_image.shape[1] / image.shape[1]
        yScale = cropped_image.shape[0] / image.shape[0]

        # Draw the hand annotations on the image (and scale the result to undo the cropping).
        if results.multi_hand_landmarks:
            # Loop through every hand
            for landmarks in results.multi_hand_landmarks:
                # Scale every point down to the appropriate size, and add the offset of the starting point
                for point in landmarks.landmark:
                    point.x = (point.x * xScale) + \
                        (start_crop[0] / image.shape[1])
                    point.y = (point.y * yScale) + \
                        (start_crop[1] / image.shape[0])
                # Draw all the landmarks for this hand
                mp_drawing.draw_landmarks(
                    image, landmarks, mp_hands.HAND_CONNECTIONS)
            return True, results.multi_hand_landmarks, image
    return False, None, image
