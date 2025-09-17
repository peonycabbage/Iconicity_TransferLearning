import cv2
import mediapipe as mp
import numpy as np
import os
import argparse

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


def extract_keypoints(results):
    """Extract left/right hand, shoulders, and wrists as flattened vector."""
    lh = (
        np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
        if results.left_hand_landmarks else np.zeros(21 * 3)
    )
    rh = (
        np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks else np.zeros(21 * 3)
    )

    if results.pose_landmarks:
        ls = np.array([
            results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x,
            results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y,
            results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].z,
        ])
        rs = np.array([
            results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x,
            results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y,
            results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].z,
        ])
        lw = np.array([
            results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].x,
            results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].y,
            results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].z,
        ])
        rw = np.array([
            results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].x,
            results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].y,
            results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].z,
        ])
    else:
        ls = rs = lw = rw = np.zeros(3)

    return np.concatenate([lh, rh, ls, rs, lw, rw])


def process_frames(input_root: str, output_root: str):
    """
    Walk input_root/category/video/frame.png, extract keypoints, save to output_root.
    """
    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        refine_face_landmarks=True
    ) as holistic:

        for category in os.listdir(input_root):
            category_path = os.path.join(input_root, category)
            if not os.path.isdir(category_path):
                continue
            print(f"Processing category: {category_path}")

            for video in os.listdir(category_path):
                video_path = os.path.join(category_path, video)
                if not os.path.isdir(video_path):
                    continue

                for filename in os.listdir(video_path):
                    frame_path = os.path.join(video_path, filename)
                    image = cv2.imread(frame_path)
                    if image is None:
                        continue

                    results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                    coordspath = os.path.join(output_root, category, video)
                    os.makedirs(coordspath, exist_ok=True)

                    np.save(
                        os.path.join(coordspath, os.path.splitext(filename)[0]),
                        extract_keypoints(results).astype(np.float32)
                    )

                print(f"  -> saved skeletons for video {video}")


def main():
    parser = argparse.ArgumentParser(description="Convert video frames to skeleton keypoints with MediaPipe Holistic.")
    parser.add_argument("--input-dir", required=True,
                        help="Root folder containing extracted video frames (category/video/frames).")
    parser.add_argument("--output-dir", required=True,
                        help="Folder to save skeleton numpy files (mirrors input structure).")
    args = parser.parse_args()

    process_frames(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
