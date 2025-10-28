import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

finger_tips = [4, 8, 12, 16, 20]

palette = [
    (255, 255, 255),  # blanc
    (0, 0, 255),      # rouge
    (0, 255, 0),      # vert
    (255, 0, 0),      # bleu
    (0, 255, 255),    # jaune cyan
    (0, 128, 255),    # orange foncé
    (30, 30, 30)      # gris très foncé (à la place du noir)
]
draw_color = palette[0]  # blanc par défaut

square_size = 50
margin = 10
palette_bg_color = (200, 200, 200)  # gris clair

cv2.namedWindow("Hand Sign Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Hand Sign Detection", 720, 600)

cap = cv2.VideoCapture(0)
canvas = None
last_point = None
drawing = False

def fingers_up(hand_landmarks):
    fingers = []
    # Pouce (x)
    if hand_landmarks.landmark[finger_tips[0]].x < hand_landmarks.landmark[finger_tips[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)
    # Autres doigts (y)
    for tip in finger_tips[1:]:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

def draw_palette(frame):
    x0 = margin
    y0 = margin
    width = len(palette)*(square_size + margin) + margin
    height = square_size + 2*margin
    # Fond gris clair pour palette
    cv2.rectangle(frame, (x0 - margin, y0 - margin), (x0 + width, y0 + height), palette_bg_color, -1)
    for i, color in enumerate(palette):
        x = x0 + i * (square_size + margin)
        cv2.rectangle(frame, (x, y0), (x + square_size, y0 + square_size), color, -1)
        cv2.rectangle(frame, (x, y0), (x + square_size, y0 + square_size), (255, 255, 255), 2)
    return x0, y0

def select_color(x, y, x0, y0):
    for i in range(len(palette)):
        px = x0 + i * (square_size + margin)
        if px <= x <= px + square_size and y0 <= y <= y0 + square_size:
            return palette[i]
    return None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    h, w, _ = frame.shape

    x0, y0 = draw_palette(frame)

    gomme_active = False

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        fingers = fingers_up(hand_landmarks)
        total_fingers = sum(fingers)

        x_index = int(hand_landmarks.landmark[8].x * w)
        y_index = int(hand_landmarks.landmark[8].y * h)

        selected = select_color(x_index, y_index, x0, y0)
        if selected is not None:
            draw_color = selected

        # Dessiner : pouce + index levés
        if fingers[0] == 1 and fingers[1] == 1 and total_fingers == 2:
            drawing = True
            if last_point is not None:
                cv2.line(canvas, last_point, (x_index, y_index), draw_color, 5)
            cv2.circle(frame, (x_index, y_index), 10, draw_color, cv2.FILLED)
            last_point = (x_index, y_index)
        else:
            drawing = False
            last_point = None

        # Gomme : main complètement ouverte
        if total_fingers == 5:
            erase_radius = 40
            cv2.circle(canvas, (x_index, y_index), erase_radius, (0, 0, 0), -1)
            gomme_active = True

    

    else:
        drawing = False
        last_point = None

    # Afficher gomme active sous la palette
    if gomme_active:
        text_pos_y = y0 + square_size + 2*margin + 20
        cv2.putText(frame, "GOMME ACTIVE", (margin, text_pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

    # Dessiner la toile directement sur la frame (opaque)
    mask = canvas.astype(bool)
    frame[mask] = canvas[mask]

    # Afficher couleur sélectionnée à droite de la palette
    color_x = x0 + len(palette) * (square_size + margin)
    cv2.rectangle(frame, (color_x, y0), (color_x + square_size, y0 + square_size), draw_color, -1)
    cv2.rectangle(frame, (color_x, y0), (color_x + square_size, y0 + square_size), (255, 255, 255), 2)

    cv2.imshow("Hand Sign Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        canvas = np.zeros_like(frame)
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
