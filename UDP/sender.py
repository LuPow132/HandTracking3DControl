import cv2
import mediapipe as mp
import math
import socket 
import json 

# --- 1. UDP CONNECTION SETUP ---
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# Sender socket does not bind; it just sends to the receiver's address
serverAddressPort = ("127.0.0.1", 5052) # Localhost, Port 5052 (Blender Listener)

# --- CONFIGURATION ---
mp_hands = mp.solutions.hands
# max_num_hands is set to 1 as per your Second Code for simplicity
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- DEPTH CALIBRATION ---
FOCAL_LENGTH = 500 
REAL_HAND_SIZE_CM = 9 

def calculate_distance(p1, p2):
    return math.hypot(p2.x - p1.x, p2.y - p1.y)

def track_finger(img, landmarks, tip_id):
    """Checks if a finger is extended and draws a circle."""
    comparison_joint_id = tip_id - 2
    wrist = landmarks[0]
    tip = landmarks[tip_id]
    joint = landmarks[comparison_joint_id]

    dist_wrist_to_tip = calculate_distance(wrist, tip)
    dist_wrist_to_joint = calculate_distance(wrist, joint)

    if dist_wrist_to_tip > dist_wrist_to_joint:
        h, w, c = img.shape
        cx, cy = int(tip.x * w), int(tip.y * h)
        cv2.circle(img, (cx, cy), 15, (255, 255, 255), 2)

def draw_line(img, landmarks, tip_id1, tip_id2):
    h, w, c = img.shape
    x1, y1 = int(landmarks[tip_id1].x * w), int(landmarks[tip_id1].y * h)
    x2, y2 = int(landmarks[tip_id2].x * w), int(landmarks[tip_id2].y * h)
    cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)

def get_wrist_rotation(img, landmarks):
    """Calculates the 'Roll' of the wrist."""
    h, w, c = img.shape
    p5 = landmarks[5]
    p17 = landmarks[17]
    x1, y1 = p5.x * w, p5.y * h
    x2, y2 = p17.x * w, p17.y * h

    angle_rad = math.atan2(y2 - y1, x2 - x1)
    angle_deg = math.degrees(angle_rad)
    
    cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 3)
    return int(angle_deg)


# --- MAIN LOOP ---
cap = cv2.VideoCapture(0)
print("Press 'q' to exit.")

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = hand_landmarks.landmark
            h, w, c = img.shape

            # --- VISUAL TRACKING (from Second Code) ---
            track_finger(img, lm_list, 8) 
            track_finger(img, lm_list, 4) 
            draw_line(img, lm_list, 8, 4)
            
            # --- CALCULATIONS ---
            rotation_angle = get_wrist_rotation(img, lm_list)

            # Finger Distance (Pinch)
            x1, y1 = int(lm_list[8].x * w), int(lm_list[8].y * h)
            x2, y2 = int(lm_list[4].x * w), int(lm_list[4].y * h)
            pinch_dist_px = math.hypot(x2 - x1, y2 - y1)

            # Depth Estimation
            wrist = lm_list[0]
            index_mcp = lm_list[5]
            
            palm_size_px = math.hypot((index_mcp.x * w) - (wrist.x * w), 
                                      (index_mcp.y * h) - (wrist.y * h))
            
            if palm_size_px > 0:
                depth_cm = (REAL_HAND_SIZE_CM * FOCAL_LENGTH) / palm_size_px
            else:
                depth_cm = 0

            # --- UDP DATA PREPARATION (from First Code) ---
            # Data is rounded for cleaner transmission
            hand_data = {
                "angle": rotation_angle,
                "pinch": round(pinch_dist_px, 2),
                "x": round(lm_list[0].x, 2), # Normalized Wrist X
                "y": round(lm_list[0].y, 2), # Normalized Wrist Y
                # Note: You can send depth_cm here too if needed for Blender's Z translation
                "depth": round(depth_cm, 2) 
            }
            
            # --- UDP SEND ---
            try:
                sock.sendto(str.encode(json.dumps(hand_data)), serverAddressPort)
            except Exception as e:
                # This will catch errors if the port is busy or network fails
                # print(f"UDP Send Error: {e}") 
                pass 

            # --- DISPLAY LOGIC (from Second Code) ---
            dynamic_scale = palm_size_px * 0.005
            dynamic_scale = max(0.5, dynamic_scale) 
            
            wrist_x = int(lm_list[0].x * w)
            wrist_y = int(lm_list[0].y * h)
            line_height = int(30 * dynamic_scale)

            # Line 1: Angle
            cv2.putText(img, f"Angle: {rotation_angle} deg", (wrist_x, wrist_y),
                        cv2.FONT_HERSHEY_SIMPLEX, dynamic_scale, (255, 255, 255), 2)
            
            # Line 2: Pinch Distance
            cv2.putText(img, f"Pinch: {int(pinch_dist_px)} px", (wrist_x, wrist_y + line_height),
                        cv2.FONT_HERSHEY_SIMPLEX, dynamic_scale, (255, 255, 255), 2)
            
            # Line 3: Depth (Z)
            cv2.putText(img, f"Depth: {int(depth_cm)} cm", (wrist_x, wrist_y + (line_height * 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, dynamic_scale, (0, 255, 0), 2)

    cv2.imshow("Hand Tracking", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
