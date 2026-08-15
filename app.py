import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a,b,c):
    a = np.array(a) #肩
    b = np.array(b) #ひじ
    c = np.array(c) #手首
    #正しいベクトル計算による角度算出
    radians = np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
        return angle
    
st.title("バスケ シュートフォース分析AI")
st.subheader("〜クレイ・トンプソン度 判定システム（AIコーチング版）〜")

uploaded_file = st.file_upload("動画ファイルを選択してください...",type=["mp4","mov","avi"])

if uploaded_file is not None:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())

    cap = cv2.VideoCapture("temp_video.mp4")
    st_frame = st.empty()

    #各フェーズのデータを記録する変数
    setup_elbow_angle = None
    release_elbow_angle = None
    follow_through_frames = 0

    #時系列解析用の状態管理フラグ
    phase = "SETUP_WAITING" #SETUP_WAITING -> SETTING_UP -> RELEASING -> FOLLOW_THROUGH -> FINISHED

    prev_wrist_y = None
    min_wrist_y = 1.0 #画像座標系では、上に行くほどyは小さくなる(0が最上部)
    highest_wrist_frame = 0
    frame_count = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True

            display_status = "Analyzing..."

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNEXTIONS)

                try:
                    landmarks = results.pose_landmarks.landmark
                    h,w,_ = image.shape

                    #1.利き腕の判定（手首が上がっている方を基準とする）
                    is_right_handed = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y <landmarks[mp_pose.PoseLandmark.LEFT.value].y

                    if is_right_handed:
                        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                        elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value.y]]
                        wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    else:
                        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value.y]]
                        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                        elbow_angle = calculate_angle(shoulder, elbow, wrist)

                        #2. 各フェーズの自動検知ロジック
                        current_wrist_y = wrist[1]

                        #[フェーズ1: セットアップ検知]
                        #手首が肩より上にあり、かつ肘がしっかりと曲がっている瞬間を捉える
                        if phase == "SETUP_WATING" and current_wrist_y < shoulder[1] and elbow_angle <120:
                            phase = "SETTING_UP"
                            setup_elbow_angle = elbow_angle #セットアップ時のタメの角度を記録

                        #[フェーズ2: リリース検知]
                        #セットアップ後、手首が上昇(yが減少)し続け、肘が伸び始めた瞬間
                        if phase == "SETTING_UP":
                            if prev_wrist_y is not None and current_wrist_y < prev_wrist_y:
                                #最も高い位置(yが最小)を更新し続ける
                                if current_wrist_y < min_wrist_y:
                                    min_wrist_y = current_wrist_y
                                    release_elbow_angle = elbow_angle #最高点付近の肘の角度

                        #手首の上昇が止まり、肘がほぼ伸びきったらリリースされたとみなす
                        if elbow_angle > 150:
                            phase = "RELEASING"

                        #[フェーズ3: フォロースルー検知]
                        #リリース後、高い位置で肘が伸びた状態(160度以上)を何フレーム維持できるか
                        if phase == "RELEASING":
                            if elbow_angle > 160:
                                follow_throught_frames += 1
                                display_status = "Follow Through..."
                            else:
                                #腕を下げたら計測終了
                                if follow_through_frames > 5:
                                    phase = "FINISHED"
                                    prev_wrist_y = current_wrist_y

                                    #画面へのリアルタイムステータス表示
                                    cv2.putTest(image, f"Status: {phase}",(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2, cv2.LINE_AA)
                                    cv2.putTest(image, f"Elbow: {int(elbow_angle)}deg",(30,90), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2, cv2.LINE_AA)

                except Exception as e:
                    pass

                st_frame.image(image, channels="RGB")

            cap.release()

        #3. クレイ・トンプソンの理想値と比較した詳細なAIコーチング評価
        st.success("シュートフォームの分析が完了しました!")

        #理想値(クレイ・トンプソンのフォームをベースとした基準)
        #セットアップ：約90〜100度、リリース：約170〜180度、フォロースルー維持：10フレーム以上(約0.3秒以上)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(label="①セットアップ時の肘角度", value=f"{int(setup_elbow_angle)}° "if setup_elbow_angle else "未検知")
            if setup_elbow_angle:
                if 85 <= setup_elbow_angle <= 105:
                    st.write("🟢**完璧です！**クレイのように綺麗な直角(90°付近)でボールをタメられています。力強いシュートが打てます。")
                elif setup_elbow_angle <85:
                    st.write("🟡**やや鋭角です。**懐が狭くなっているため、打点が高くなりにくい可能性があります。もう少し肘をゆったり構えましょう。")
                else:
                    st.write("🟡**肘が開き気味です。**シュートの力がボールに伝わりにくくなります。脇を締め、約90度を意識してください。")

        with col2:
            st.metric(label="②リリース時の肘角度", value=f"{int(release_elbow_angle)}° " if release_elbow_angle else "未検知")
            if release_elbow_angle:
                if release_elbow_angle >= 165:
                    st.write("🟢**素晴らしい！**肘が最後まで真っ直ぐ伸びきっています。シュートのブレが最小限に抑えられます。")
                else:
                    st.write("🔴**押し出しが不十分です。**肘が曲がったままリリースしているため、飛距離が出にくく、軌道が安定しません。しっかり空へ向かって腕を伸ばしましょう。")

        with col3:
            #フレーム数を秒数に換算(動画が30fpsの場合)
            duration = follow_through_frames / 30.0
            st.metric(label="③フォロースルー維持", value=f"{duration:.2f}秒")
            if follow_through_frames >= 12:
                st.write("🟢**素晴らしい残心です！**打った後も腕を綺麗に残せています。これがクレイ・トンプソン並みの高い確率の秘訣です。")
            elif 5<= follow_through_frames < 12:
                st.write("🟡**あと少し！**悪くはないですが、シュート直後に腕をすぐ下げてしまう癖があります。ボールがリングに届くまでポーズをキープしましょう。")
            else:
                st.write("🔴**フォロースルーがありません。**シュートの手をすぐ引いてしまうと、指先のコントロールが狂いやすくなります。")


            #総合スコアの算出
            if setup_elbow_angle and release_elbow_angle:
                score_setup = max(0, 100 - abs(95 - setup_elbow_angle)* 2)
                score_release = max(0, 100 - abs(175 - release_elbow_angle)* 3)
                score_follow = min(100, follow_through_frames * 7)

                final_score = (score_setup + score_release + score_follow) / 3
                st.subheader(f"🔥総合クレイ・トンプソン度:{int(final_score)}% 🔥")





