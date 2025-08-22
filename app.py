# app.py

import os
import time
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# Important: import mediapipe after Streamlit sets up environment to avoid partial import issues
import mediapipe as mp

# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(page_title="Dance Clone", layout="wide")

st.title("🤖💃 Dance Clone Interactive")
st.markdown(
    """
This app creates a real-time mirrored neon stick-figure clone using your webcam, powered by MediaPipe Pose and OpenCV.
- Click START below and allow camera access.
- Move in front of the camera to see your cyberpunk clone.
- A live 3D pose visualization updates on the right.
    """
)

# ---------------------------
# Pose connections (MediaPipe enums)
# ---------------------------
POSE_CONNECTIONS_MP_ENUM = frozenset([
    (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER),
    (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.LEFT_ELBOW),
    (mp.solutions.pose.PoseLandmark.LEFT_ELBOW, mp.solutions.pose.PoseLandmark.LEFT_WRIST),

    (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_ELBOW),
    (mp.solutions.pose.PoseLandmark.RIGHT_ELBOW, mp.solutions.pose.PoseLandmark.RIGHT_WRIST),

    (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.LEFT_HIP),
    (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_HIP),
    (mp.solutions.pose.PoseLandmark.LEFT_HIP, mp.solutions.pose.PoseLandmark.RIGHT_HIP),

    (mp.solutions.pose.PoseLandmark.LEFT_HIP, mp.solutions.pose.PoseLandmark.LEFT_KNEE),
    (mp.solutions.pose.PoseLandmark.LEFT_KNEE, mp.solutions.pose.PoseLandmark.LEFT_ANKLE),

    (mp.solutions.pose.PoseLandmark.RIGHT_HIP, mp.solutions.pose.PoseLandmark.RIGHT_KNEE),
    (mp.solutions.pose.PoseLandmark.RIGHT_KNEE, mp.solutions.pose.PoseLandmark.RIGHT_ANKLE),
])

# ---------------------------
# UI controls
# ---------------------------
with st.sidebar:
    st.header("Settings")
    color_interval = st.slider("Color change interval (sec)", 1, 10, 3, 1)
    glow_thick = st.slider("Glow line thickness", 2, 20, 10, 1)
    glow_joint_radius = st.slider("Glow joint radius", 4, 20, 8, 1)
    clone_offset = st.slider("Clone horizontal offset (px)", -400, 400, 0, 10)
    model_complexity = st.selectbox("Model complexity", [0, 1, 2], index=1)
    min_det_conf = st.slider("Min detection confidence", 0.1, 1.0, 0.6, 0.05)
    min_track_conf = st.slider("Min tracking confidence", 0.1, 1.0, 0.6, 0.05)
    smooth_landmarks = st.checkbox("Smooth landmarks", True)

# Neon palette (BGR for OpenCV)
GLOW_COLORS_RGB = [
    (0, 255, 255),    # Cyan
    (255, 0, 255),    # Magenta
    (255, 255, 0),    # Yellow
    (0, 255, 0),      # Lime
    (255, 105, 180),  # Hot Pink
]

# ---------------------------
# Helpers
# ---------------------------
def get_dynamic_color(start_time, color_list, interval_seconds):
    elapsed = time.time() - start_time
    idx = int(elapsed // interval_seconds) % len(color_list)
    return color_list[idx]

def draw_glowing_stick_figure(
    image_bgr,
    pose_landmarks,
    connections,
    color_bgr,
    frame_w,
    frame_h,
    offset_x_px=0,
    joint_radius=8,
    line_thick=10
):
    if not pose_landmarks:
        return

    overlay = np.zeros_like(image_bgr, dtype=np.uint8)
    pts = {}

    for idx, lm in enumerate(pose_landmarks.landmark):
        if lm.visibility < 0.3:
            continue

        x_px = int(lm.x * frame_w)
        y_px = int(lm.y * frame_h)

        # Mirror around center
        mirrored_x = frame_w - x_px
        final_x = mirrored_x + offset_x_px

        pts[idx] = (final_x, y_px)

        if 0 <= final_x < frame_w and 0 <= y_px < frame_h:
            cv2.circle(overlay, (final_x, y_px), joint_radius, color_bgr, -1)
            cv2.circle(overlay, (final_x, y_px), joint_radius + 4, color_bgr, 2)

    if connections:
        for a, b in connections:
            a_idx = a.value
            b_idx = b.value
            if a_idx in pts and b_idx in pts:
                p1 = pts[a_idx]
                p2 = pts[b_idx]
                cv2.line(overlay, p1, p2, color_bgr, line_thick, cv2.LINE_AA)
                # brighter inner line
                inner = (
                    min(255, color_bgr[0] + 50),
                    min(255, color_bgr[1] + 50),
                    min(255, color_bgr[2] + 50),
                )
                cv2.line(overlay, p1, p2, inner, max(1, line_thick // 2), cv2.LINE_AA)

    image_bgr[:] = cv2.add(image_bgr, overlay)

# ---------------------------
# WebRTC Transformer
# ---------------------------
mp_pose = mp.solutions.pose

class DanceCloneTransformer(VideoTransformerBase):
    def __init__(self):
        self.pose = mp_pose.Pose(
            min_detection_confidence=min_det_conf,
            min_tracking_confidence=min_track_conf,
            smooth_landmarks=smooth_landmarks,
            model_complexity=model_complexity,
        )
        if "color_start_time" not in st.session_state:
            st.session_state.color_start_time = time.time()
        # placeholder for sending landmarks to plotting area
        st.session_state.current_landmarks = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape

        # Run pose
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self.pose.process(rgb)
        rgb.flags.writeable = True

        # Color
        color = get_dynamic_color(
            st.session_state.color_start_time,
            GLOW_COLORS_RGB,
            color_interval
        )
        # Draw clone
        if res.pose_landmarks:
            draw_glowing_stick_figure(
                img,
                res.pose_landmarks,
                POSE_CONNECTIONS_MP_ENUM,
                color,
                w, h,
                offset_x_px=clone_offset,
                joint_radius=glow_joint_radius,
                line_thick=glow_thick
            )
            st.session_state.current_landmarks = res.pose_landmarks
        else:
            st.session_state.current_landmarks = None

        return img

    def __del__(self):
        if hasattr(self, "pose") and self.pose:
            self.pose.close()

# ---------------------------
# Layout: Video left, 3D plot right
# ---------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Webcam & Clone")
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    ctx = webrtc_streamer(
        key="dance-clone",
        video_transformer_factory=DanceCloneTransformer,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    if not ctx.state.playing:
        st.info("Click START and allow camera access.")
    else:
        st.success("Webcam processing active!")

with col2:
    st.subheader("3D Pose Visualization")
    placeholder = st.empty()

    # Keep a persistent figure
    if "fig3d" not in st.session_state:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title("3D Pose Estimation")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.view_init(elev=20., azim=-35)
        st.session_state.fig3d = fig

    # Render current landmarks if available
    if st.session_state.get("current_landmarks", None):
        fig = st.session_state.fig3d
        ax = fig.axes[0]
        ax.clear()
        ax.set_title("3D Pose Estimation")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.view_init(elev=20., azim=-35)

        lm = st.session_state.current_landmarks
        xs = [p.x for p in lm.landmark]
        ys = [-p.y for p in lm.landmark]
        zs = [-p.z for p in lm.landmark]

        ax.scatter(xs, ys, zs, c="red", s=15)

        for a, b in POSE_CONNECTIONS_MP_ENUM:
            pa = lm.landmark[a.value]
            pb = lm.landmark[b.value]
            ax.plot([pa.x, pb.x], [-pa.y, -pb.y], [-pa.z, -pb.z], "blue", linewidth=2)

        placeholder.pyplot(fig)
    else:
        fig = st.session_state.fig3d
        ax = fig.axes[0]
        ax.clear()
        ax.set_title("3D Pose Estimation (Waiting...)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.view_init(elev=20., azim=-35)
        placeholder.pyplot(fig)

st.markdown("---")
st.caption("Inspired by MediaPipe, OpenCV, and Streamlit WebRTC.")
