import asyncio
import cv2
import mediapipe as mp
import websockets

FLIP_HORIZONTAL = 1
WINDOW_NAME = 'GestureCapture'
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600


async def producer_handler(websocket):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

    def calculate_distance(landmark1, landmark2):
        return np.sqrt(
            (landmark1.x - landmark2.x) ** 2 + (landmark1.y - landmark2.y) ** 2 + (landmark1.z - landmark2.z) ** 2)

    def recognize_index_finger_click_gesture(hand_landmarks):
        index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        return calculate_distance(index_finger_tip, thumb_tip) < 0.05

    def recognize_middle_finger_click_gesture(hand_landmarks):
        middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        return calculate_distance(middle_finger_tip, thumb_tip) < 0.05

    mp_drawing = mp.solutions.drawing_utils

    # cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)

    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, FLIP_HORIZONTAL)
        result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if result.multi_hand_landmarks and result.multi_handedness:
            left_hand_index_finger_click = False
            right_hand_index_finger_click = False

            left_hand_middle_finger_click = False
            right_hand_middle_finger_click = False

            for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                hand_label = handedness.classification[0].label

                is_index_finger_clicked = recognize_index_finger_click_gesture(hand_landmarks)
                if is_index_finger_clicked:
                    if hand_label == "Left":
                        left_hand_index_finger_click = is_index_finger_clicked
                    else:
                        right_hand_index_finger_click = is_index_finger_clicked

                is_middle_finger_clicked = recognize_middle_finger_click_gesture(hand_landmarks)
                if is_middle_finger_clicked:
                    if hand_label == "Left":
                        left_hand_middle_finger_click = is_middle_finger_clicked
                    else:
                        right_hand_middle_finger_click = is_middle_finger_clicked

            if left_hand_index_finger_click and right_hand_index_finger_click:
                await websocket.send("Rotate")
                await asyncio.sleep(0.2)
            elif left_hand_index_finger_click:
                await websocket.send("Left")
            elif right_hand_index_finger_click:
                await websocket.send("Right")
            elif left_hand_middle_finger_click and right_hand_middle_finger_click:
                await websocket.send("Drop")
                await asyncio.sleep(0.2)

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


async def consumer_handler(websocket):
    async for message in websocket:
        print(f"Message received from client: {message}")
        await websocket.send(f"Echo: {message}")


async def handler(websocket, path):
    consumer_task = asyncio.create_task(consumer_handler(websocket))
    producer_task = asyncio.create_task(producer_handler(websocket))
    done, pending = await asyncio.wait([consumer_task, producer_task], return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()


async def main():
    async with websockets.serve(handler, "localhost", 8765):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
