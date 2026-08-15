import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile

# ページ基本設定
st.set_page_config(page_title="AIバスケシュート分析ツール", page_icon="🏀", layout="wide")

# MediaPipeの初期化
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    """肩(a)・肘(b)・手首(c)の3点から肘の角度を計算する関数"""
    a = np.array(a)  # 肩
    b = np.array(b)  # 肘
    c = np.array(c)  # 手首
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

st.title("🏀 AIバスケシュート分析 (OpenCV + MediaPipe)")
st.markdown("動画をアップロードすると、AIが骨格を検知して**「肘の角度」**をリアルタイムで自動計算します。")

# サイドバー設定
st.sidebar.header("📁 設定 & アップロード")
uploaded_file = st.sidebar.file_uploader("シュート動画をアップロード (.mp4, .mov)", type=["mp4", "mov"])
detection_confidence = st.sidebar.slider("検知信頼度 (Confidence)", 0.0, 1.0, 0.5)

# メインコンテンツ
col1, col2 = st.columns([2, 1])

if uploaded_file is not None:
    # 一時ファイルに動画を書き出し
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    
    # 描画用プレースホルダー
    with col1:
        st.subheader("📹 骨格検知フィード")
        frame_placeholder = st.empty()
        
    with col2:
        st.subheader("📊 リアルタイム分析データ")
        angle_metric = st.empty()
        advice_box = st.empty()
        
    # メインループ
    with mp_pose.Pose(min_detection_confidence=detection_confidence, min_tracking_confidence=detection_confidence) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # メモリ効率と処理速度のためにリサイズ (Streamlit表示用)
            frame = cv2.resize(frame, (640, 480))
            
            # MediaPipe用にRGBに変換
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # ポーズ検知の実行
            results = pose.process(image)
            
            # 再び描画用にwritableに戻す
            image.flags.writeable = True
            
            current_angle = None
            
            # 骨格が検知された場合の処理
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # 右利きのシューターを想定（肩・肘・手首のインデックス）
                # 実際の運用では左右どちらがカメラに近いかを自動判定するとより実用的です
                shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                
                # 角度計算
                current_angle = calculate_angle(shoulder, elbow, wrist)
                
                # 画面上に角度テキストを描画 (OpenCV)
                h, w, _ = image.shape
                elbow_pixel = (int(elbow[0] * w), int(elbow[1] * h))
                cv2.putText(image, f"{int(current_angle)} deg", 
                            elbow_pixel, 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                
                # 骨格ラインの描画
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Streamlit画面の更新
            frame_placeholder.image(image, channels="RGB")
            
            if current_angle is not None:
                angle_metric.metric(label="現在の右肘の角度", value=f"{int(current_angle)}°")
                
                # 簡易的なシュートフォームアドバイス
                if current_angle < 80:
                    advice_box.warning("💡 アドバイス: 肘が畳まれすぎています。セットアップ時に約90度を意識すると力が伝わりやすくなります。")
                elif 80 <= current_angle <= 100:
                    advice_box.success("🎯 アドバイス: 理想的な肘のタメ（約90度）が作れています！そのまま連動させてリリースしましょう。")
                else:
                    advice_box.info("💡 アドバイス: 肘が伸び気味です。ボールを呼び込む際に、手首と肘の連動を意識してください。")
                    
    cap.release()
    st.success("動画の解析が完了しました！")

else:
    with col1:
        st.info("サイドバーから動画ファイルをアップロードすると、ここに骨格検知の映像が表示されます。")
    with col2:
        st.write("※動画解析が始まると、ここにリアルタイムの角度データとフォーム判定アドバイスが表示されます。")
