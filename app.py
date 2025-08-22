import os
import time
import math
import tempfile
from typing import Tuple, List, Optional

import streamlit as st
import numpy as np
import cv2

import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# -----------------------------
# UI CONFIG
# -----------------------------
st.set_page_config(page_title="💃🕺 Dance Clone Interactive", layout="wide")

# Title and instructions
st.title("🤖 🕺 Dance Clone Interactive")
st.write(
    "This app creates a real-time mirrored neon stick-figure clone using your webcam or a video file, "
    "powered by MediaPipe Pose and OpenCV."
)
st.write(
    "- Click START below and allow camera access for live mode, or switch to the Video tab to process a file.\n"
    "- Move in front of the camera to see your cyberpunk clone.\n"
    "- A live 3D pose visualization is shown on the right (in live mode)."
)

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Settings")

    color_change_interval = st.slider("Color change interval (sec)", 1, 10, 3)
    glow_thickness = st.slider("Glow line thickness", 1, 20, 10)
    joint_radius = st.slider("Glow joint radius", 2, 20, 8)
    clone_offset_x = st.slider("Clone horizontal offset (px)", -400, 400, 0)

    model_complexity = st.selectbox("Model complexity", [0, 1, 2], index=1)

    det_conf = st.slider("Min detection confidence", 0.10, 1.00, 0.60, step=0.01)
    trk_conf = st.slider("Min tracking confidence", 0.10, 1.00, 0.60, step=0.01)
    smooth_lms = st.checkbox("Smooth landmarks", value=True)


# -----------------------------
# MediaPipe helpers
# -----------------------------
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

POSE_CONNECTIONS = list(mp_pose.POSE_CONNECTIONS)


def _cycle_color(t_seconds: float, period: float) -> Tuple[int, int, int]:
    """Cycle through bright neon-like colors over time."""
    # Map time to a color wheel
    phase = (t_seconds % period) / period  # 0-1
    # Create RGB from three shifted sinusoids for neon palette
    r = int(127.5 * (1 + math.sin(2 * math.pi * (phase))))
    g = int(127.5 * (1 + math.sin(2 * math.pi * (phase + 1 / 3))))
    b = int(127.5 * (1 + math.sin(2 * math.pi * (phase + 2 / 3))))
    # Boost saturation
    r = min(255, int(r * 1.2))
    g = min(255, int(g * 1.2))
    b = min(255, int(b * 1.2))
    return (b, g, r)  # OpenCV uses BGR


def _draw_neon_stick_figure(
    frame_bgr: np.ndarray,
    landmarks: List[Tuple[int, int]],
    color: Tuple[int, int, int],
    glow_thickness: int,
    joint_radius: int,
):
    """Draw a glowing stick figure given landmark pixel coordinates."""
    h, w = frame_bgr.shape[:2]

    # Soft glow by drawing thicker translucent layers first
    overlay = frame_bgr.copy()

    # Draw connections
    for (i1, i2) in POSE_CONNECTIONS:
        if i1 < len(landmarks) and i2 < len(landmarks):
            p1 = landmarks[i1]
            p2 = landmarks[i2]
            if p1 is not None and p2 is not None:
                cv2.line(overlay, p1, p2, color, thickness=glow_thickness, lineType=cv2.LINE_AA)

    # Draw joints
    for p in landmarks:
        if p is not None:
            cv2.circle(overlay, p, joint_radius, color, -1, lineType=cv2.LINE_AA)

    # Blend overlay for glow effect
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0, frame_bgr)

    # Sharper bright pass
    for (i1, i2) in POSE_CONNECTIONS:
        if i1 < len(landmarks) and i2 < len(landmarks):
            p1 = landmarks[i1]
            p2 = landmarks[i2]
            if p1 is not None and p2 is not None:
                cv2.line(frame_bgr, p1, p2, color, thickness=max(1, glow_thickness // 2), lineType=cv2.LINE_AA)
    for p in landmarks:
        if p is not None:
            cv2.circle(frame_bgr, p, max(1, joint_radius // 2), (255, 255, 255), -1, lineType=cv2.LINE_AA)


def _extract_landmarks_px(results, width: int, height: int) -> List[Optional[Tuple[int, int]]]:
    """Convert MediaPipe normalized landmarks to pixel coordinates; returns list indexed by landmark id."""
    if not results.pose_landmarks:
        return [None] * 33
    pts = []
    for lm in results.pose_landmarks.landmark:
        if lm.visibility is not None and lm.visibility < 0.2:
            pts.append(None)
            continue
        x = int(lm.x * width)
        y = int(lm.y * height)
        if 0 <= x < width and 0 <= y < height:
            pts.append((x, y))
        else:
            pts.append(None)
    # Ensure length 33
    if len(pts) < 33:
        pts += [None] * (33 - len(pts))
    return pts


def render_neon_clone_frame(
    frame_bgr: np.ndarray,
    pose_processor: mp_pose.Pose,
    now_seconds: float,
    color_period: float,
    glow_thick: int,
    joint_rad: int,
    clone_offset_x: int,
) -> np.ndarray:
    """Run pose, draw neon stick figure and its mirrored clone."""
    # Work copy
    frame = frame_bgr.copy()
    h, w = frame.shape[:2]

    # Pose inference expects RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_processor.process(rgb)

    # Extract pixel landmarks
    pts = _extract_landmarks_px(results, w, h)
    if all(p is None for p in pts):
        return frame  # nothing to draw

    # Time-based neon color
    color = _cycle_color(now_seconds, color_period)

    # Draw on main subject
    _draw_neon_stick_figure(frame, pts, color, glow_thick, joint_rad)

    # Create mirrored clone horizontally and offset
    # Flip X across image center
    mirrored = []
    for p in pts:
        if p is None:
            mirrored.append(None)
        else:
            x, y = p
            mx = w - x  # mirror around center
            mirrored.append((mx, y))

    clone = frame.copy()
    _draw_neon_stick_figure(clone, mirrored, color, glow_thick, joint_rad)

    # Shift clone horizontally
    M = np.float32([[1, 0, clone_offset_x], [0, 1, 0]])
    shifted = cv2.warpAffine(clone, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)

    # Blend clone onto original
    blended = cv2.addWeighted(frame, 1.0, shifted, 0.6, 0)
    return blended


# -----------------------------
# Live webcam (streamlit-webrtc)
# -----------------------------
class NeonCloneProcessor(VideoTransformerBase):
    def __init__(self, config):
        self.cfg = config
        self.pose = mp_pose.Pose(
            model_complexity=self.cfg["model_complexity"],
            min_detection_confidence=self.cfg["det_conf"],
            min_tracking_confidence=self.cfg["trk_conf"],
            smooth_landmarks=self.cfg["smooth_lms"],
        )
        # For deterministic color when using frame count, we still rely on real time for live
        self.t0 = time.time()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        now_s = time.time() - self.t0
        out = render_neon_clone_frame(
            img,
            self.pose,
            now_seconds=now_s,
            color_period=self.cfg["color_period"],
            glow_thick=self.cfg["glow_thickness"],
            joint_rad=self.cfg["joint_radius"],
            clone_offset_x=self.cfg["clone_offset_x"],
        )
        return av.VideoFrame.from_ndarray(out, format="bgr24")


# -----------------------------
# Layout: Tabs for Live / Video
# -----------------------------
tab_live, tab_video = st.tabs(["Webcam", "Video file"])

with tab_live:
    st.subheader("Live Webcam & Clone")

    live_cfg = {
        "model_complexity": model_complexity,
        "det_conf": det_conf,
        "trk_conf": trk_conf,
        "smooth_lms": smooth_lms,
        "glow_thickness": glow_thickness,
        "joint_radius": joint_radius,
        "clone_offset_x": clone_offset_x,
        "color_period": float(color_change_interval),
    }

    ctx = webrtc_streamer(
        key="dance-clone-live",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=lambda: NeonCloneProcessor(live_cfg),
        async_processing=True,
    )

    if ctx.state.playing:
        st.success("Webcam processing active!")

with tab_video:
    st.subheader("Process a video file")

    up = st.file_uploader("Upload a video", type=["mp4", "mov", "webm", "mkv"])
    max_seconds = st.slider("Process first N seconds (to keep it quick)", 5, 180, 60)
    downscale_to = st.selectbox("Max output size", ["Original", "1280x720", "854x480"], index=1)

    process_btn = st.button("Process Video", disabled=up is None)

    if process_btn and up:
        # Save upload to a temp file
        src_suffix = os.path.splitext(up.name)[1] or ".mp4"
        src_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=src_suffix)
        src_tmp.write(up.read())
        src_tmp.close()

        cap = cv2.VideoCapture(src_tmp.name)
        if not cap.isOpened():
            st.error("Cannot open the uploaded video.")
        else:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Resolve output size
            if downscale_to == "1280x720":
                out_w, out_h = 1280, 720
            elif downscale_to == "854x480":
                out_w, out_h = 854, 480
            else:
                out_w, out_h = src_w, src_h

            # Preserve aspect ratio when downscaling
            if (out_w, out_h) != (src_w, src_h):
                scale = min(out_w / src_w, out_h / src_h)
                out_w = int(src_w * scale)
                out_h = int(src_h * scale)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))

            # Build a CPU pose processor
            with mp_pose.Pose(
                model_complexity=model_complexity,
                min_detection_confidence=det_conf,
                min_tracking_confidence=trk_conf,
                smooth_landmarks=smooth_lms,
            ) as pose:
                # progress
                max_frames_by_time = int(max_seconds * fps)
                max_frames = min(total_frames if total_frames > 0 else max_frames_by_time, max_frames_by_time)
                prog = st.progress(0.0)
                status = st.empty()

                for i in range(max_frames):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if (src_w, src_h) != (out_w, out_h):
                        frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

                    # Use deterministic time based on frame index for stable color cycling
                    now_s = i / fps
                    out = render_neon_clone_frame(
                        frame,
                        pose,
                        now_seconds=now_s,
                        color_period=float(color_change_interval),
                        glow_thick=glow_thickness,
                        joint_rad=joint_radius,
                        clone_offset_x=clone_offset_x,
                    )

                    writer.write(out)

                    if max_frames > 0:
                        prog.progress(min(0.999, (i + 1) / max_frames))
                    status.text(f"Processing frame {i+1}/{max_frames} ...")

            cap.release()
            writer.release()

            st.success("Done! Preview below:")
            with open(out_path, "rb") as f:
                video_bytes = f.read()
                st.video(video_bytes)

            st.download_button(
                "Download processed video",
                data=video_bytes,
                file_name="dance_clone_processed.mp4",
                mime="video/mp4",
            )

            # Cleanup the uploaded file (keep output until session ends)
            try:
                os.unlink(src_tmp.name)
            except Exception:
                pass


# -----------------------------
# Footer
# -----------------------------
st.write("---")
st.caption("Inspired by MediaPipe, OpenCV, and Streamlit WebRTC.")
