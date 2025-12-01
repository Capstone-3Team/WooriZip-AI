import cv2
import mediapipe as mp
import numpy as np

# ============================================
# 0. Landmark 실패 카운터 (전역 유지)
# ============================================
FAILED_LANDMARK_FRAMES = 0
FAILED_THRESHOLD = 3   # 3프레임 연속 실패 시 come_in


# ============================================
# 1. Mediapipe FaceMesh 초기화
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
# 3. 상태 판단 함수 + landmark 실패 누적 처리
# ============================================
def analyze_face_from_frame(frame):
    global FAILED_LANDMARK_FRAMES

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    # ---------------------------
    # 0. landmark 실패 처리
    # ---------------------------
    if not results.multi_face_landmarks:
        FAILED_LANDMARK_FRAMES += 1

        # 🔥 3프레임 연속 실패일 때만 come_in 출력
        if FAILED_LANDMARK_FRAMES >= FAILED_THRESHOLD:
            return {
                "state": "come_in",
                "message": "화면 안으로 들어오세요",
                "is_good": False
            }
        else:
            # perfect 유지 (잠깐 흔들려도 오류 안 띄움)
            return {"state": "perfect", "message": "", "is_good": True}

    # landmark 감지 성공 → 실패 카운트 초기화
    FAILED_LANDMARK_FRAMES = 0

    face = results.multi_face_landmarks[0]

    xs = [lm.x for lm in face.landmark]
    ys = [lm.y for lm in face.landmark]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    bw = max_x - min_x
    bh = max_y - min_y

    # ---------------------------
    # 1) 이목구비 가려질 때 perfect 처리 (come_in 억제)
    # ---------------------------
    if not facial_features_visible(face):
        return {"state": "perfect", "message": "", "is_good": True}

    # ---------------------------
    # 2) 너무 가까움 → move_back
    # ---------------------------
    if bw > 0.70 or bh > 0.70:
        return {
            "state": "move_back",
            "message": "조금 뒤로 물러나세요",
            "is_good": False
        }

    # ---------------------------
    # 3) 얼굴 노출 비율 체크 (완화: 0.3)
    # ---------------------------
    vis_x0 = np.clip(min_x, 0, 1); vis_x1 = np.clip(max_x, 0, 1)
    vis_y0 = np.clip(min_y, 0, 1); vis_y1 = np.clip(max_y, 0, 1)

    vis_w = (vis_x1 - vis_x0) / bw if bw > 0 else 0
    vis_h = (vis_y_1 - vis_y0) / bh if bh > 0 else 0
    visible_ratio = min(vis_w, vis_h)

    if visible_ratio < 0.3:  # 30% 이하만 come_in
        return {
            "state": "come_in",
            "message": "화면 안으로 들어오세요",
            "is_good": False
        }

    # ---------------------------
    # 4) 눈 높이 체크
    # ---------------------------
    eye_ids = [33, 133, 362, 263]
    eye_ys = [face.landmark[i].y for i in eye_ids]
    avg_eye_y = sum(eye_ys) / len(eye_ys)

    if avg_eye_y < 0.15:
        return {
            "state": "come_in",
            "message": "화면 안으로 들어오세요",
            "is_good": False
        }

    # ---------------------------
    # 정상 상태
    # ---------------------------
    return {
        "state": "perfect",
        "message": "",
        "is_good": True
    }
