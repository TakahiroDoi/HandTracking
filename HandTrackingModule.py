import cv2  # OpenCV
import mediapipe as mp  # for hand landmark detection


class HandDetector:

    def __init__(self, mode=False, max_hands=2, detection_conf=0.5, track_conf=0.5):
        self.mode = mode
        self.maxHands = max_hands
        self.detectionConf = detection_conf
        self.trackConf = track_conf

        self.mpDraw = mp.solutions.drawing_utils
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionConf, self.trackConf)

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_number=0, draw=True):
        """
        Find specified hand's landmarks using self.results

        :param img: image
        :param hand_number: hand number, left or right
        :param draw: True to draw big circles (manually coded)
        :return: no return
        """

        print_landmark_position = False
        list_pos = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_number]
            for idx, lm in enumerate(my_hand.landmark):
                # lm.x and .y are window normalized position so getting pixel coord in (cx, cy)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                list_pos.append([idx, cx, cy])
                if print_landmark_position:
                    print(idx, cx, cy)
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return list_pos
