import cv2
import mediapipe as mp
import time

external_monitor_to_right = True

ind_camera = 0
cap = cv2.VideoCapture(ind_camera)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

cv2.namedWindow('Image')

# Show it on the external display to the right
if external_monitor_to_right:
    cv2.moveWindow('Image', 1440, 0)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # for idx, lm in enumerate(handLms.landmark):
                # print(idx)
                # print(lm)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # Show fps
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)

    cv2.imshow("Image", img)

    windowPosition = cv2.getWindowImageRect('Image')
    print(windowPosition)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        # waitKey(1) was too short.
        break

cv2.destroyAllWindows()
cap.release()
