import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

# MediaPipeの初期化
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# クレイ・トンプソンのフォームベンチマーク（理想値）
KLAY_ELBOW_TARGET = 90.0
KLAY_HIP_FLEX_MAX = 140.0
ANGLE_TOLERANCE = 10.0


def calculate_angle(a, b, c):
    """3点の座標から角度（度数法）を計算"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c - b, c - b) - np.arctan2(
        a - b, a - b
    )
    angle = np.abs(radians * 180.0 / np.pi)
    return 360.0 - angle if angle > 180.0 else angle


st.title("🏀 クレイ・トンプソン基準・シュートフォーム分析")
st.write(
    "NBA最高峰のキャッチ＆シューター、クレイ・トンプソンの美しいメカニクスとあなたのフォームを比較します。"
)

# --- 🛠 サイドバーの設定 ---
st.sidebar.header("⚙️ 設定・ステータス")

# 利き手の選択モード（自動判定 or 手動上書き）
hand_mode = st.sidebar.radio(
    "利き手の手動選択", ["自動判定", "右利き", "左利き"]
)

# ステータス表示用のプレースホルダー
hand_indicator = st.sidebar.empty()
status_elbow = st.sidebar.empty()
status_hip = st.sidebar.empty()
status_balance = st.sidebar.empty()

# 動画のアップロード
uploaded_file = st.file_uploader(
    "シュート動画（横、または斜め前からのアングルを推奨）", type=["mp4", "mov"]
)

if uploaded_file is not None:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())

    cap = cv2.VideoCapture("temp_video.mp4")
    frame_placeholder = st.empty()

    # 自動判定用の変数を初期化
    detected_hand = None

    with mp_pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # --- 🎯 利き手決定ロジック ---
                if hand_mode == "自動判定":
                    # 最初の数フレームで位置が高い（y座標が小さい）手首を判定・ロックする
                    if detected_hand is None:
                        right_wrist_y = landmarks[
                            mp_pose.PoseLandmark.RIGHT_WRIST.value
                        ].y
                        left_wrist_y = landmarks[
                            mp_pose.PoseLandmark.LEFT_WRIST.value
                        ].y
                        detected_hand = (
                            "RIGHT"
                            if right_wrist_y < left_wrist_y
                            else "LEFT"
                        )
                    current_hand = detected_hand
                    hand_indicator.info(
                        f"🤖 判定された利き手: {'右投げ' if current_hand == 'RIGHT' else '左投げ'}"
                    )
                elif hand_mode == "右利き":
                    current_hand = "RIGHT"
                    hand_indicator.success("✋ 手動設定: 右投げ")
                else:
                    current_hand = "LEFT"
                    hand_indicator.success("✋ 手動設定: 左投げ")

                # --- 🦴 選択された手に基づく座標抽出 ---
                if current_hand == "RIGHT":
                    shoulder = [
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                    ]
                    elbow = [
                        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
                    ]
                    wrist = [
                        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
                    ]
                    hip = [
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                    ]
                    knee = [
                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
                    ]
                else:
                    shoulder = [
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                    ]
                    elbow = [
                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                    ]
                    wrist = [
                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
                    ]
                    hip = [
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                    ]
                    knee = [
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
                    ]

                # --- 📊 トンプソン基準の計算と表示 ---
                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                hip_angle = calculate_angle(shoulder, hip, knee)
                torso_lean = abs(shoulder[0] - hip[0]) * 100

                # 肘の判定
                if (
                    abs(elbow_angle - KLAY_ELBOW_TARGET)
                    <= ANGLE_TOLERANCE
                ):
                    status_elbow.success(
                        f"肘の角度: {int(elbow_angle)}° (トンプソン級! 90°)"
                    )
                    color = (0, 255, 0)
                else:
                    status_elbow.warning(
                        f"肘の角度: {int(elbow_angle)}° (目標 90°)"
                    )
                    color = (255, 255, 255)

                # 股関節（タメ）の判定
                if hip_angle < KLAY_HIP_FLEX_MAX:
                    status_hip.success(
                        f"ディップ: {int(hip_angle)}° (十分なタメ)"
                    )
                else:
                    status_hip.info(
                        f"ディップ: {int(hip_angle)}° (もう少し腰を落とせます)"
                    )

                # 体幹ブレ判定
                if torso_lean < 5.0:
                    status_balance.success("軸のブレ: なし (垂直を維持)")
                else:
                    status_balance.error("軸のブレ: あり (体幹を意識)")

                # 骨格とテキストの描画
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )
                h, w, _ = image.shape
                cv2.putText(
                    image,
                    f"Elbow: {int(elbow_angle)} deg",
                    tuple(np.multiply(elbow, [w, h]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                )

            frame_placeholder.image(image, channels="RGB", use_container_width=True)

    cap.release()
    st.success("すべての分析が完了しました！")
