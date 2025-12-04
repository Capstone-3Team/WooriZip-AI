import cv2
import mediapipe as mp
import numpy as np
import os
# ============================================
# 0. Landmark 실패 카운터 + 마지막 상태
# ============================================
FAILED_LANDMARK_FRAMES = 0
FAILED_THRESHOLD = 3
LAST_STATE = "perfect"

# ============================================
# 1. FaceMesh 초기화
# ============================================
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

mp_face_mesh = mp.solutions.face_mesh

mesh_detector = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,  # GPU 사용되는 옵션 → 반드시 False
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)



# ============================================
# 2. 이목구비 체크
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
# 3. 얼굴 크기 기반 필터링 (멀리 있는 사람 제거)
# ============================================
def filter_front_faces(faces):
    """
    faces 중 area >= 0.05 인 얼굴만 반환 (카메라 가까이 있는 얼굴만 사용)
    """
    front_faces = []
    for face in faces:
        xs = [lm.x for lm in face.landmark]
        ys = [lm.y for lm in face.landmark]
        bw = max(xs) - min(xs)  # bounding width
        bh = max(ys) - min(ys)  # bounding height
        area = bw * bh

        if area >= 0.05:       # 🔥 이 값 이하이면 배경 사람
            front_faces.append(face)

    return front_faces


# ============================================
# 4. 얼굴 상태 분석 (여러 명 처리)
# ============================================
def analyze_face(face):
    """
    하나의 얼굴에 대해 perfect/come_in/move_back 판정
    """
    xs = [lm.x for lm in face.landmark]
    ys = [lm.y for lm in face.landmark]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    bw = max_x - min_x
    bh = max_y - min_y

    # 이목구비 가려지면 perfect 유지
    if not facial_features_visible(face):
        return "perfect"

    # 너무 가까움
    if bw > 0.70 or bh > 0.70:
        return "move_back"

    # 노출 비율
    vis_x0 = np.clip(min_x, 0, 1); vis_x1 = np.clip(max_x, 0, 1)
    vis_y0 = np.clip(min_y, 0, 1); vis_y1 = np.clip(max_y, 0, 1)

    vis_w = (vis_x1 - vis_x0) / bw if bw > 0 else 0
    vis_h = (vis_y1 - vis_y0) / bh if bh > 0 else 0
    visible_ratio = min(vis_w, vis_h)

    if visible_ratio < 0.3:
        return "come_in"

    # 눈 높이
    eye_ids = [33, 133, 362, 263]
    eye_ys = [face.landmark[i].y for i in eye_ids]
    avg_eye_y = sum(eye_ys) / len(eye_ys)
    if avg_eye_y < 0.15:
        return "come_in"

    return "perfect"


# ============================================
# 5. 메인 함수 (여러 얼굴 처리)
# ============================================
def analyze_face_from_frame(frame):
    global FAILED_LANDMARK_FRAMES, LAST_STATE

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    # 0) landmark 실패 → idle 또는 come_in
    if not results.multi_face_landmarks:
        FAILED_LANDMARK_FRAMES += 1

        if FAILED_LANDMARK_FRAMES >= FAILED_THRESHOLD:
            LAST_STATE = "come_in"
            return {"state": "come_in", "message": "화면 안으로 들어오세요", "is_good": False}

        return {"state": "idle", "message": "", "is_good": False}

    # 성공 → 실패 카운트 초기화
    FAILED_LANDMARK_FRAMES = 0

    # 🔥 여러 얼굴 중 배경 인물 제거
    front_faces = filter_front_faces(results.multi_face_landmarks)

    # 전경 얼굴이 하나도 없으면 idle 처리
    if len(front_faces) == 0:
        return {"state": "idle", "message": "", "is_good": False}

    # 🔥 여러 얼굴 있을 때 규칙:
    # 하나라도 come_in → come_in
    # 모두 move_back → move_back
    # 그 외 → perfect
    states = [analyze_face(face) for face in front_faces]

    if "come_in" in states:
        LAST_STATE = "come_in"
        return {"state": "come_in", "message": "화면 안으로 들어오세요", "is_good": False}

    if all(s == "move_back" for s in states):
        LAST_STATE = "move_back"
        return {"state": "move_back", "message": "조금 뒤로 물러나세요", "is_good": False}

    # 전경에 perfect가 하나라도 있으면 perfect
    LAST_STATE = "perfect"
    return {"state": "perfect", "message": "", "is_good": True}
