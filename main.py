#!/usr/bin/env python
"""
Example script to use HandTrackingModule.
Hand(s) is detected in the captured image, and the landmark locations are visualized and returned.

In the pop up window, fps of tracking is shown on the upper left.
Press "q" to quit.
"""

import cv2
import time
import HandTrackingModule as htm

external_monitor_to_right = False
print_window_position = False

# Show it on the external display to the right
if external_monitor_to_right:
    window_offset_x = 1440
else:
    window_offset_x = 0
cv2.namedWindow('Image', cv2.WINDOW_AUTOSIZE)
cv2.moveWindow('Image', window_offset_x, 0)

prev_time, curr_time = 0, 0

ind_camera = 0
cap = cv2.VideoCapture(ind_camera)
detector = htm.HandDetector()

while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    list_pos = detector.find_position(img, draw=True)
    if len(list_pos) == 0:
        pass
    else:
        print(list_pos[0])

    # Show fps
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)

    cv2.imshow("Image", img)

    if print_window_position:
        windowPosition = cv2.getWindowImageRect('Image')
        print(windowPosition)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        # waitKey(1) was too short.
        break

cv2.destroyAllWindows()
cap.release()