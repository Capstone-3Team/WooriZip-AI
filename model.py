# model.py
import cv2
import mediapipe as mp
import numpy as np

# ============================================
# 1. Mediapipe FaceMesh 초기화
# ============================================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ============================================
# 2. 손으로 얼굴 일부 가린 경우 처리 (come_in 방지)
# ============================================
def facial_features_visible(face):
    """
    nose(1), left eye(33), right eye(263), mouth(13)
    4개 중 3개 이상 보이면 정상으로 간주
    """
    key_ids = [1, 33, 263, 13]
    visible = 0

    for idx in key_ids:
        lm = face.landmark[idx]
        if 0 < lm.x < 1 and 0 < lm.y < 1:
            visible += 1

    return visible >= 3


# ============================================
# 3. 메인 얼굴 분석 로직 (app.py에서 호출)
# ============================================
def analyze_face_from_frame(frame):
    """
    입력: OpenCV BGR 프레임
    출력:
        {
            "message": "...",
            "state": "perfect" / "come_in" / "move_back",
            "is_good": bool
        }
    """
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    # 얼굴 없음 → come_in
    if not results.multi_face_landmarks:
        return {
            "message": "화면 안으로 들어오세요",
            "state": "come_in",
            "is_good": False
        }

    face = results.multi_face_landmarks[0]

    xs = [lm.x for lm in face.landmark]
    ys = [lm.y for lm in face.landmark]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    bw = max_x - min_x  # 얼굴 width ratio
    bh = max_y - min_y  # 얼굴 height ratio

    # ===============================
    # 0) 손으로 얼굴 일부 가린 경우 → perfect 처리
    # ===============================
    if not facial_features_visible(face):
        return {
            "message": "완벽합니다!",
            "state": "perfect",
            "is_good": True
        }

    # ===============================
    # 1) 너무 가까움 → move_back
    # ===============================
    if bw > 0.70 or bh > 0.70:
        return {
            "message": "조금 뒤로 물러나세요",
            "state": "move_back",
            "is_good": False
        }

    # ===============================
    # 2) 얼굴 보이는 비율 계산 (프레임 밖으로 나간 정도)
    # ===============================
    vis_x0 = np.clip(min_x, 0, 1)
    vis_x1 = np.clip(max_x, 0, 1)
    vis_y0 = np.clip(min_y, 0, 1)
    vis_y1 = np.clip(max_y, 0, 1)

    vis_w = (vis_x1 - vis_x0) / bw if bw > 0 else 0
    vis_h = (vis_y1 - vis_y0) / bh if bh > 0 else 0
    visible_ratio = min(vis_w, vis_h)

    if visible_ratio < 0.5:
        return {
            "message": "화면 안으로 들어오세요",
            "state": "come_in",
            "is_good": False
        }

    # ===============================
    # 3) 눈 위치가 너무 위 → come_in
    # ===============================
    eye_ids = [33, 133, 362, 263]
    eye_ys = [face.landmark[i].y for i in eye_ids]
    avg_eye_y = sum(eye_ys) / len(eye_ys)

    if avg_eye_y < 0.15:
        return {
            "message": "화면 안으로 들어오세요",
            "state": "come_in",
            "is_good": False
        }

    # ===============================
    # 4) 정상!
    # ===============================
    return {
        "message": "완벽합니다!",
        "state": "perfect",
        "is_good": True
    }
