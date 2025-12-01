import cv2
import mediapipe as mp
import numpy as np

# ============================================
# 1. Mediapipe FaceMesh 초기화 (전역 1회만)
# ============================================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ============================================
# 2. 이목구비 체크 (손 가림 방지)
# ============================================
def facial_features_visible(face):
    key_ids = [1, 33, 263, 13]  # 코, 왼눈, 오른눈, 입
    visible = 0

    for idx in key_ids:
        lm = face.landmark[idx]
        if 0 <= lm.x <= 1 and 0 <= lm.y <= 1:
            visible += 1

    return visible >= 3


# ============================================
# 3. 상태 판단 함수
# ============================================
def analyze_face_from_frame(frame):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return {"state": "come_in", "message": "화면 안으로 들어오세요", "is_good": False}

    face = results.multi_face_landmarks[0]

    xs = [lm.x for lm in face.landmark]
    ys = [lm.y for lm in face.landmark]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    bw = max_x - min_x
    bh = max_y - min_y

    # ---------------------------
    # 0) 이목구비가 안 보이면 perfect 취급
    # ---------------------------
    if not facial_features_visible(face):
        return {"state": "perfect", "message": "", "is_good": True}

    # ---------------------------
    # 1) 너무 가까움 → move_back
    # ---------------------------
    if bw > 0.70 or bh > 0.70:
        return {
            "state": "move_back",
            "message": "조금 뒤로 물러나세요",
            "is_good": False
        }

    # ---------------------------
    # 2) 얼굴 보이는 영역 비율 체크
    # ---------------------------
    vis_x0 = np.clip(min_x, 0, 1); vis_x1 = np.clip(max_x, 0, 1)
    vis_y0 = np.clip(min_y, 0, 1); vis_y1 = np.clip(max_y, 0, 1)

    vis_w = (vis_x1 - vis_x0) / bw if bw > 0 else 0
    vis_h = (vis_y1 - vis_y0) / bh if bh > 0 else 0
    visible_ratio = min(vis_w, vis_h)

    if visible_ratio < 0.3:  # 30% 이하만 경고
    return {
        "state": "come_in",
        "message": "화면 안으로 들어오세요",
        "is_good": False
    }

    # ---------------------------
    # 3) 눈 위치 체크
    # ---------------------------
    eye_ids = [33, 133, 362, 263]
    eye_ys = [face.landmark[i].y for i in eye_ids]
    avg_eye_y = sum(eye_ys) / len(eye_ys)

    if avg_eye_y < 0.15:
        return {"state": "come_in", "message": "화면 안으로 들어오세요", "is_good": False}

    # ---------------------------
    # 정상 상태
    # ---------------------------
    return {"state": "perfect", "message": "", "is_good": True}
