import cv2
import mediapipe as mp
import time

external_monitor_to_right = False
print_window_position = False

ind_camera = 0
cap = cv2.VideoCapture(ind_camera)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime, cTime = 0, 0

# Show it on the external display to the right
if external_monitor_to_right:
    window_offset_x = 1440
else:
    window_offset_x = 0

cv2.namedWindow('Image', cv2.WINDOW_AUTOSIZE)
cv2.moveWindow('Image', window_offset_x, 0)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for idx, lm in enumerate(handLms.landmark):
                # lm.x and .y are window normalized position so getting pixel coord in (cx, cy)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(idx, cx, cy)
                if idx == 15:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # Show fps
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)

    cv2.imshow("Image", img)

    if print_window_position:
        windowPosition = cv2.getWindowImageRect('Image')
        print(windowPosition)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        # waitKey(1) was too short.
        break

cv2.destroyAllWindows()
cap.release()
