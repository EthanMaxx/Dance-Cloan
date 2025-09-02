# dance.py  — improved, drop-in replacement for your original file
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

# Pose connections (same indices you used — valid for MediaPipe Pose's 33 landmarks)
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15),  # Left Arm
    (12, 14), (14, 16),            # Right Arm
    (11, 23), (12, 24),            # Torso
    (23, 24), (23, 25), (24, 26),  # Hips
    (25, 27), (26, 28),            # Thighs
    (27, 31), (28, 32)             # Calves
]


def clamp_point(pt, w, h):
    """Clamp (x,y) so drawing calls don't get crazy coordinates."""
    x = int(max(0, min(w - 1, int(pt[0]))))
    y = int(max(0, min(h - 1, int(pt[1]))))
    return x, y


def normalize_for_plot(lm):
    """
    Turn MediaPipe's normalized landmark into a centered coordinate set
    that sits roughly in [-1,1] for nicer 3D plotting.
    """
    x = (lm.x - 0.5) * 2
    y = (0.5 - lm.y) * 2  # invert Y so plot feels like a human view
    z = lm.z * 2
    return x, y, z


def main(cam_index=0, clone_offset=None):
    # Colors: OpenCV uses BGR. I converted your RGB list to BGR for expected colors.
    colors_bgr = [
        (0, 0, 255),   # red
        (0, 255, 0),   # green
        (255, 0, 0),   # blue
        (0, 255, 255), # yellow (BGR)
        (255, 255, 0)  # cyan (BGR)
    ]
    color_index = 0
    last_color_change = time.time()

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"Error: cannot open camera index {cam_index}. Exiting.")
        sys.exit(1)

    # Matplotlib interactive 3D figure (kept small for performance)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    plt.ion()

    # Use MediaPipe Pose as a context manager so resources are cleaned up
    with mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Warning: empty frame received. Exiting loop.")
                    break

                h, w, _ = frame.shape
                if clone_offset is None:
                    clone_offset = w // 4
                center_x = w // 2

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)

                # rotate glow color every 3 seconds
                if time.time() - last_color_change > 3:
                    color_index = (color_index + 1) % len(colors_bgr)
                    last_color_change = time.time()
                glow_color = colors_bgr[color_index]

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    landmarks_2d = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
                    mirror_2d = [(2 * center_x - x + clone_offset, y) for x, y in landmarks_2d]
                    mirror_2d = [clamp_point(p, w, h) for p in mirror_2d]
                    landmarks_3d = [normalize_for_plot(lm) for lm in landmarks]

                    glow_layer = np.zeros_like(frame, dtype=np.uint8)

                    # draw thick colored lines (glow) + thinner white core
                    for i, j in POSE_CONNECTIONS:
                        p1 = mirror_2d[i]
                        p2 = mirror_2d[j]
                        cv2.line(glow_layer, p1, p2, glow_color, thickness=10, lineType=cv2.LINE_AA)
                        cv2.line(glow_layer, p1, p2, (255, 255, 255), thickness=3, lineType=cv2.LINE_AA)

                    # draw joints
                    for x, y in mirror_2d:
                        cv2.circle(glow_layer, (int(x), int(y)), 10, glow_color, -1, lineType=cv2.LINE_AA)
                        cv2.circle(glow_layer, (int(x), int(y)), 5, (255, 255, 255), -1, lineType=cv2.LINE_AA)

                    frame = cv2.addWeighted(frame, 0.8, glow_layer, 0.6, 0)

                    # update 3D plot (small pause for responsiveness)
                    ax.clear()
                    ax.set_xlim([-1, 1])
                    ax.set_ylim([-1, 1])
                    ax.set_zlim([-1, 1])
                    ax.set_title("3D Pose (normalized)")
                    for i, j in POSE_CONNECTIONS:
                        x_vals = [landmarks_3d[i][0], landmarks_3d[j][0]]
                        y_vals = [landmarks_3d[i][1], landmarks_3d[j][1]]
                        z_vals = [landmarks_3d[i][2], landmarks_3d[j][2]]
                        ax.plot(x_vals, y_vals, z_vals, marker="o")
                    plt.draw()
                    plt.pause(0.001)

                cv2.imshow("Cyberpunk Dance Clone", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        except KeyboardInterrupt:
            print("Interrupted by user.")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            plt.ioff()
            plt.close(fig)


if __name__ == "__main__":
    main()
