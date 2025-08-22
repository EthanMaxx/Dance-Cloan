# app.py

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# --- Configuration & Constants ---
# (Same as your Colab notebook for consistency)
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

# For drawing, it's easier to use the integer indices if POSE_CONNECTIONS was originally tuples of ints
# Let's define it based on the landmark indices (0-32) as per original README if needed,
# or map the enums if preferred for drawing. For simplicity, we'll stick to the enums for MediaPipe processing
# and handle drawing connections carefully. The provided POSE_CONNECTIONS from README seems to use indices.
# MediaPipe's PoseLandmark enum values are the indices.
# (11, 12) -> (LEFT_SHOULDER, RIGHT_SHOULDER)
# (11, 13) -> (LEFT_SHOULDER, LEFT_ELBOW)
# (13, 15) -> (LEFT_ELBOW, LEFT_WRIST)
# ... and so on. The POSE_CONNECTIONS_MP_ENUM above is correct.

GLOW_COLORS_RGB = [
    (0, 255, 255),  # Cyan
    (255, 0, 255),  # Magenta
    (255, 255, 0),  # Yellow
    (0, 255, 0),    # Lime
    (255, 105, 180),# Hot Pink
]
COLOR_TRANSITION_INTERVAL = 3  # seconds
SMOOTH_LANDMARKS = True

# --- MediaPipe Initialization ---
mp_pose = mp.solutions.pose
# Note: pose_instance will be created within the VideoTransformer class or context

# --- Helper Functions (adapted for Streamlit) ---
def get_dynamic_color(start_time, color_list, interval):
    elapsed_time = time.time() - start_time
    color_index = int(elapsed_time / interval) % len(color_list)
    return color_list[color_index]

def draw_glowing_stick_figure(image, landmarks_mp, connections, color, frame_width, frame_height, offset_x=0):
    if not landmarks_mp:
        return

    overlay = np.zeros_like(image, dtype=np.uint8)
    landmark_points = {}

    for idx, landmark in enumerate(landmarks_mp.landmark):
        if landmark.visibility < 0.3: # Skip less visible landmarks
            continue
        cx_orig, cy_orig = int(landmark.x * frame_width), int(landmark.y * frame_height)

        # Mirroring for the clone:
        mirrored_cx = frame_width - cx_orig
        final_cx = mirrored_cx + offset_x # Additional offset for positioning clone

        landmark_points[idx] = (final_cx, cy_orig)

        if 0 <= final_cx < frame_width and 0 <= cy_orig < frame_height:
            cv2.circle(overlay, (final_cx, cy_orig), 8, color, -1)
            cv2.circle(overlay, (final_cx, cy_orig), 12, color, 2)

    if connections:
        for connection in connections:
            start_node_idx = connection[0].value # Get integer index from enum
            end_node_idx = connection[1].value   # Get integer index from enum

            if start_node_idx in landmark_points and end_node_idx in landmark_points:
                start_point = landmark_points[start_node_idx]
                end_point = landmark_points[end_node_idx]
                cv2.line(overlay, start_point, end_point, color, 10, cv2.LINE_AA)
                cv2.line(overlay, start_point, end_point, (min(255,color[0]+50), min(255,color[1]+50), min(255,color[2]+50)), 5, cv2.LINE_AA)

    image[:] = cv2.add(image, overlay)


# --- 3D Plotting Setup ---
# fig_3d_main, ax_3d_main = plt.subplots(subplot_kw={'projection': '3d'}) # Create once globally
# Using st.pyplot, it's better to create the figure inside the function or pass it around if needed.

def plot_3d_pose_st(landmarks_mp, connections, existing_fig=None):
    if not landmarks_mp:
        return None

    if existing_fig:
        fig = existing_fig
        ax = fig.axes[0]
        ax.clear() # Clear previous frame
    else:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Pose Estimation')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.view_init(elev=20., azim=-35) # Consistent view

    xs = [lm.x for lm in landmarks_mp.landmark]
    ys = [-lm.y for lm in landmarks_mp.landmark] # Invert Y for typical 3D plot
    zs = [-lm.z for lm in landmarks_mp.landmark] # Invert Z

    ax.scatter(xs, ys, zs, c='red', marker='o', s=15)

    if connections:
        for connection in connections:
            start_lm = landmarks_mp.landmark[connection[0].value]
            end_lm = landmarks_mp.landmark[connection[1].value]
            ax.plot([start_lm.x, end_lm.x],
                    [-start_lm.y, -end_lm.y],
                    [-start_lm.z, -end_lm.z],
                    'blue', linewidth=2)
    return fig


# --- Streamlit WEBRTC Video Transformer ---
class DanceCloneTransformer(VideoTransformerBase):
    def __init__(self):
        self.pose_instance = mp_pose.Pose(
            min_detection_confidence=0.6, # Increased slightly
            min_tracking_confidence=0.6,  # Increased slightly
            smooth_landmarks=SMOOTH_LANDMARKS,
            model_complexity=1
        )
        if "color_start_time" not in st.session_state:
            st.session_state.color_start_time = time.time()
        self.fig_3d = None # To hold the figure object for reuse

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_height, img_width, _ = img.shape
        output_frame = img.copy()

        # Process with MediaPipe
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.pose_instance.process(rgb_frame)
        rgb_frame.flags.writeable = True

        # Get dynamic color
        current_glow_color = get_dynamic_color(st.session_state.color_start_time, GLOW_COLORS_RGB, COLOR_TRANSITION_INTERVAL)

        # Draw original landmarks (optional, can be turned off for clarity)
        # if results.pose_landmarks:
        #     mp.solutions.drawing_utils.draw_landmarks(
        #         output_frame, results.pose_landmarks, POSE_CONNECTIONS_MP_ENUM,
        #         landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(245,117,66), thickness=1, circle_radius=1),
        #         connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(245,66,230), thickness=1)
        #     )

        # Draw glowing clone
        if results.pose_landmarks:
            # The clone is mirrored. `offset_x=0` means it's on the opposite side.
            # If you want to push it further to one side (e.g. to make space for original),
            # you can adjust this. For now, 0 should provide a good mirrored effect.
            draw_glowing_stick_figure(output_frame, results.pose_landmarks, POSE_CONNECTIONS_MP_ENUM,
                                      current_glow_color, img_width, img_height, offset_x=0)

        # Store landmarks for 3D plot (if Streamlit can access this outside transform)
        # A bit tricky with streamlit-webrtc class structure for external updates.
        # We'll update a session state variable.
        if results.pose_landmarks:
            st.session_state.current_landmarks = results.pose_landmarks
        else:
            st.session_state.current_landmarks = None

        return output_frame

    def __del__(self): # Cleanup
        if hasattr(self, 'pose_instance') and self.pose_instance:
            self.pose_instance.close()

# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="Dance Clone")

st.title("🤖💃 Dance Clone Interactive")
st.markdown("""
This application uses your webcam to create a real-time interactive clone dancer beside you,
complete with glowing stick figure visuals and dynamic color transitions.
A 3D plot of your pose is also shown. **Allow webcam access when prompted.**
""")

# Initialize session state for landmarks and figure
if "current_landmarks" not in st.session_state:
    st.session_state.current_landmarks = None
if "fig_3d" not in st.session_state:
    st.session_state.fig_3d = None # Initialize figure in session state

col1, col2 = st.columns([2, 1]) # Video takes 2/3, Plot takes 1/3

with col1:
    st.subheader("Live Webcam Feed & Clone")
    # RTCConfiguration for STUN/TURN servers if needed for NAT traversal,
    # often not required for local testing or simple Streamlit Cloud deployments.
    rtc_configuration = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    webrtc_ctx = webrtc_streamer(
        key="dance-clone-streamer",
        video_transformer_factory=DanceCloneTransformer,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True, # Allow processing to not block the video stream
        # video_html_attrs={"style": "width: 100%; height: auto; border: 1px solid #ddd;"} # Example styling
    )
    if not webrtc_ctx.state.playing:
        st.info("Click 'START' to begin webcam processing.")
    else:
        st.success("Webcam processing active!")


with col2:
    st.subheader("3D Pose Visualization")
    plot_placeholder = st.empty() # Placeholder for the plot

    # This part needs to update based on the landmarks from the transformer
    # We will use a loop that checks session_state, but this is not ideal for performance
    # in Streamlit. A better way would be if the transformer could directly trigger updates.
    # For now, this will make the plot somewhat reactive.

    # We create the figure once and store it in session_state to update it
    if st.session_state.fig_3d is None:
        st.session_state.fig_3d, ax = plt.subplots(subplot_kw={'projection': '3d'})
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title('3D Pose Estimation')
        ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
        ax.view_init(elev=20., azim=-35)

    if st.session_state.current_landmarks:
        # Re-use the existing figure and axes
        fig_to_plot = st.session_state.fig_3d
        ax_to_plot = fig_to_plot.axes[0]
        ax_to_plot.clear() # Clear previous drawing
        ax_to_plot.set_xlabel('X'); ax_to_plot.set_ylabel('Y'); ax_to_plot.set_zlabel('Z')
        ax_to_plot.set_title('3D Pose Estimation')
        ax_to_plot.set_xlim([-1, 1]); ax_to_plot.set_ylim([-1, 1]); ax_to_plot.set_zlim([-1, 1])
        ax_to_plot.view_init(elev=20., azim=-35) # Re-apply view settings

        landmarks_mp = st.session_state.current_landmarks
        xs = [lm.x for lm in landmarks_mp.landmark]
        ys = [-lm.y for lm in landmarks_mp.landmark]
        zs = [-lm.z for lm in landmarks_mp.landmark]
        ax_to_plot.scatter(xs, ys, zs, c='red', marker='o', s=15)
        for connection in POSE_CONNECTIONS_MP_ENUM:
            start_lm = landmarks_mp.landmark[connection[0].value]
            end_lm = landmarks_mp.landmark[connection[1].value]
            ax_to_plot.plot([start_lm.x, end_lm.x],
                           [-start_lm.y, -end_lm.y],
                           [-start_lm.z, -end_lm.z],
                           'blue', linewidth=2)
        plot_placeholder.pyplot(fig_to_plot)
    else:
        # Display an empty or placeholder plot if no landmarks
        fig_empty, ax_empty = plt.subplots(subplot_kw={'projection': '3d'})
        ax_empty.set_xlabel('X'); ax_empty.set_ylabel('Y'); ax_empty.set_zlabel('Z')
        ax_empty.set_title('3D Pose Estimation (Waiting for data...)')
        ax_empty.set_xlim([-1, 1]); ax_empty.set_ylim([-1, 1]); ax_empty.set_zlim([-1, 1])
        plot_placeholder.pyplot(fig_empty)
        plt.close(fig_empty) # Close to free memory

st.markdown("---")
st.markdown("Inspired by [Pramzie/Dance-Clone](https://github.com/Pramzie/Dance-Clone) on GitHub.")
st.markdown("Built with [Streamlit](https://streamlit.io/), [MediaPipe](https://mediapipe.dev/), and [OpenCV](https://opencv.org/).")