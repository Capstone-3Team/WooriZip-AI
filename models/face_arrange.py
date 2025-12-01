import cv2
import mediapipe as mp
import numpy as np

# ============================================
# State smoothing 전역 변수
# ============================================
FAILED_LANDMARK_FRAMES = 0
FAILED_THRESHOLD = 3     # 3프레임 연속 landmark 실패 시 come_in
LAST_STATE = "perfect"   # 안정화된 최종 상태 저장

# ============================================
# FaceMesh 초기화
# ============================================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ============================================
# 이목구비 체크
# ============================================
def facial_features_visible(face):
    key_ids = [1, 33, 263, 13]
    visible = 0

    for idx in key_ids:
        lm = face.landmark[idx]
        if 0 <= lm.x <= 1 and 0 <= lm.y <= 1:
            visible += 1

    return visible >= 3

# ============================================
# 메인 분석 함수 (State Machine)
# ============================================
def analyze_face_from_frame(frame):
    global FAILED_LANDMARK_FRAMES, LAST_STATE

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    # ---------------------------
    # landmark 실패 → 누적 카운트 증가
    # ---------------------------
    if not results.multi_face_landmarks:
        FAILED_LANDMARK_FRAMES += 1

        if FAILED_LANDMARK_FRAMES >= FAILED_THRESHOLD:
            LAST_STATE = "come_in"
            return {
                "state": "come_in",
                "message": "화면 안으로 들어오세요",
                "is_good": False
            }
        else:
            # → 여긴 perfect 유지 (일시적 흔들림)
            return {
                "state": LAST_STATE,
                "message": "",
                "is_good": True
            }

    # landmark 감지 성공했으면 실패 카운트 초기화
    FAILED_LANDMARK_FRAMES = 0

    face = results.multi_face_landmarks[0]

    xs = [lm.x for lm in face.landmark]
    ys = [lm.y for lm in face.landmark]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    bw = max_x - min_x
    bh = max_y - min_y

    # ---------------------------
    # 1) 이목구비 가려짐 → perfect (come_in 억제)
    # ---------------------------
    if not facial_features_visible(face):
        LAST_STATE = "perfect"
        return {"state": "perfect", "message": "", "is_good": True}

    # ---------------------------
    # 2) 너무 가까움 → move_back
    # ---------------------------
    if bw > 0.70 or bh > 0.70:
        LAST_STATE = "move_back"
        return {
            "state": "move_back",
            "message": "조금 뒤로 물러나세요",
            "is_good": False
        }

    # ---------------------------
    # 3) 얼굴 노출 비율 평가 (완화: 0.3)
    # ---------------------------
    vis_x0 = np.clip(min_x, 0, 1); vis_x1 = np.clip(max_x, 0, 1)
    vis_y0 = np.clip(min_y, 0, 1); vis_y1 = np.clip(max_y, 0, 1)

    vis_w = (vis_x1 - vis_x0) / bw if bw > 0 else 0
    vis_h = (vis_y1 - vis_y0) / bh if bh > 0 else 0
    visible_ratio = min(vis_w, vis_h)

    if visible_ratio < 0.3:
        LAST_STATE = "come_in"
        return {
            "state": "come_in",
            "message": "화면 안으로 들어오세요",
            "is_good": False
        }

    # ---------------------------
    # 4) 눈 위치 체크
    # ---------------------------
    eye_ids = [33, 133, 362, 263]
    eye_ys = [face.landmark[i].y for i in eye_ids]
    avg_eye_y = sum(eye_ys) / len(eye_ys)

    if avg_eye_y < 0.15:
        LAST_STATE = "come_in"
        return {
            "state": "come_in",
            "message": "화면 안으로 들어오세요",
            "is_good": False
        }

    # ---------------------------
    # 5) 최종 정상 상태
    # ---------------------------
    LAST_STATE = "perfect"
    return {"state": "perfect", "message": "", "is_good": True}
