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
# 2. 이목구비가 프레임 안에 어느 정도 들어있는지 체크 (부분 가림 방지용)
# ============================================
def facial_features_visible(face):
    """
    코, 양 눈, 입 주변 landmark가 화면 안에 있는지 확인.
    (손으로 얼굴 일부 가렸을 때 'come_in'이 너무 빨리 뜨는 것 완화용)
    """
    key_ids = [1, 33, 263, 13]  # nose, left eye, right eye, mouth
    visible_count = 0

    for idx in key_ids:
        lm = face.landmark[idx]
        if 0 <= lm.x <= 1 and 0 <= lm.y <= 1:
            visible_count += 1

    # 4개 중 3개 이상이 프레임 안에 있으면 "이목구비가 대체로 보인다"로 간주
    return visible_count >= 3


# ============================================
# 3. 메인 얼굴 분석 로직 (웹/백엔드에서 사용)
# ============================================
def analyze_face_from_frame(frame):
    """
    입력: OpenCV BGR 프레임 (numpy array, shape: H x W x 3)
    출력(dict):
        {
            "message": "완벽합니다!" / "화면 안으로 들어오세요" / "조금 뒤로 물러나세요",
            "state": "perfect" / "come_in" / "move_back",
            "is_good": True/False
        }
    """
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    # 얼굴 없음 → 기본 안내: 들어오세요
    if not results.multi_face_landmarks:
        return {
            "message": "화면 안으로 들어오세요",
            "state": "come_in",
            "is_good": False
        }

    # 가장 먼저 감지된 얼굴만 기준으로 사용
    face = results.multi_face_landmarks[0]

    xs = [lm.x for lm in face.landmark]
    ys = [lm.y for lm in face.landmark]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    bw = max_x - min_x  # width 비율 (0~1)
    bh = max_y - min_y  # height 비율 (0~1)

    # ===============================
    # 0) 이목구비가 너무 안 보이면 → 굳이 come_in 안 띄움 (손으로 가린 경우 등)
    # ===============================
    if not facial_features_visible(face):
        # "얼굴은 있는데 포즈가 애매한 상태" → 그냥 perfect도, come_in도 아님
        return {
            "message": "완벽합니다!",
            "state": "perfect",
            "is_good": True
        }

    # ===============================
    # 1) 너무 가까움 (화면 꽉 찼음) → move_back
    # ===============================
    if bw > 0.70 or bh > 0.70:
        return {
            "message": "조금 뒤로 물러나세요",
            "state": "move_back",
            "is_good": False
        }

    # ===============================
    # 2) 얼굴 보이는 영역 비율 계산 (프레임 밖으로 나간 정도)
    # ===============================
    vis_x0 = np.clip(min_x, 0, 1); vis_x1 = np.clip(max_x, 0, 1)
    vis_y0 = np.clip(min_y, 0, 1); vis_y1 = np.clip(max_y, 0, 1)

    vis_w = (vis_x1 - vis_x0) / bw if bw > 0 else 0
    vis_h = (vis_y1 - vis_y0) / bh if bh > 0 else 0
    visible_ratio = min(vis_w, vis_h)

    # 절반 이상 프레임 밖 → come_in
    if visible_ratio < 0.5:
        return {
            "message": "화면 안으로 들어오세요",
            "state": "come_in",
            "is_good": False
        }

    # ===============================
    # 3) 눈 위치 (너무 위쪽) → 화면 아래쪽으로 내려와 달라는 의미로 come_in
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
    # 4) 정상 상태
    # ===============================
    return {
        "message": "완벽합니다!",
        "state": "perfect",
        "is_good": True
    }
