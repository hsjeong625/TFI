import cv2
import json
import os
import numpy as np
from scipy.ndimage import gaussian_filter


def draw_gaussian_heatmap(img, point, fsr_value, radius=10, intensity=1):
    """Draw a Gaussian heatmap on the image at the specified point."""
    if point is None:
        return img
    heatmap = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    x, y = round(point[0] * img.shape[1]), round(point[1] * img.shape[0])
    heatmap[y, x] = fsr_value * intensity
    heatmap = gaussian_filter(heatmap, sigma=radius)
    heatmap = np.clip((heatmap * 255).astype(np.uint8), 0, 255)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.7, heatmap_colored, 0.3, 0)
    return heatmap, overlay


def crop_frames_from_video(mp4_path, json_path, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load JSON data
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)

    # Open video file
    cap = cv2.VideoCapture(mp4_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Unable to open video file {mp4_path}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    json_list_length = len(data)

    # Verify that the number of frames matches the length of the JSON list
    if frame_count != json_list_length:
        print(f"Error: Frame count {frame_count} does not match JSON list length {json_list_length}")
        cap.release()
        return

    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index >= json_list_length:
            break

        # Get the corresponding JSON object for the current frame
        json_object = data[frame_index]
        right_points = json_object.get('right_points')
        fsr_value = json_object.get('fsr')

        if right_points and fsr_value:
            frame_height, frame_width = frame.shape[:2]
            # Extract the bounding box coordinates
            x_min = round(min(point[0] for point in right_points) * frame_width)
            y_min = round(min(point[1] for point in right_points) * frame_height)
            x_max = round(max(point[0] for point in right_points) * frame_width)
            y_max = round(max(point[1] for point in right_points) * frame_height)

            # Calculate the width and height of the box
            box_width = x_max - x_min
            box_height = y_max - y_min

            # Increase the box size by 30%
            margin_x = int(box_width * 0.2)
            margin_y = int(box_height * 0.2)

            x_min = max(x_min - margin_x, 0)
            y_min = max(y_min - margin_y, 0)
            x_max = min(x_max + margin_x, frame_width)
            y_max = min(y_max + margin_y, frame_height)

            # Crop the frame
            cropped_frame = frame[y_min:y_max, x_min:x_max]

            # Resize the cropped frame to 256x256
            resized_cropped_frame = cv2.resize(cropped_frame, (256, 256), interpolation=cv2.INTER_CUBIC)

            # Calculate the new thumb_tip position in the resized frame
            thumb_tip = right_points[4]  # Assuming the thumb tip is the 4th point in the right hand landmarks
            thumb_joint = right_points[3]
            thumb_tip_resized = (
                256 * (frame_width * (0.8 * thumb_tip[0] + 0.2 * thumb_joint[0]) - x_min) / (x_max - x_min),
                256 * (frame_height * (0.8 * thumb_tip[1] + 0.2 * thumb_joint[1]) - y_min) / (y_max - y_min)
            )

            # Save the resized cropped frame
            output_file_path = os.path.join(output_dir, f"{frame_index:04d}_({thumb_tip_resized[0]},{thumb_tip_resized[1]})_{fsr_value}.png")
            cv2.imwrite(output_file_path, resized_cropped_frame)
            # print(frame_index, mp4_path)

        frame_index += 1

    # Release the video capture object
    cap.release()

if __name__ == '__main__':

    # json_file_paths = [
    #     "data/P000, CNA, phase0, Test, trialSession-5, 180s, egocamera1_merged.json",
    #     "data/P000, CNA, phase0, Test, trialSession-5, 180s, egocamera2_merged.json",
    #     "data/P000, CNA, phase0, Test, trialSession-5, 180s, egocamera3_merged.json"
    # ]

    # output_dirs = [
    #     "data/P000, CNA, phase0, Test, trialSession-5, 180s, egocamera1_original/",
    #     "data/P000, CNA, phase0, Test, trialSession-5, 180s, egocamera2_original/",
    #     "data/P000, CNA, phase0, Test, trialSession-5, 180s, egocamera3_original/",
    # ]

    # video_paths = [
    #     "data/P000, CNA, phase0, Test, trialSession-5, 180s, egocamera1_original.mp4",
    #     "data/P000, CNA, phase0, Test, trialSession-5, 180s, egocamera2_original.mp4",
    #     "data/P000, CNA, phase0, Test, trialSession-5, 180s, egocamera3_original.mp4",
    # ]

    json_file_paths, csv_file_paths, video_paths, output_dirs = [], [], [], []
    action_label = ['CPinch','CSide1','CSide2','CSide3']
    force_label = ['Negative1','Negative2','Negative3','Positive1','Positive2','Positive3','Positive4','Positive5','Positive6']
    phase_label = ['phase0','phase1']
    for condition in action_label:
        for force in force_label[:3]:
            for k in range(1, 4):
                for j in range(1, 4):
                    json_file_path = f'data/hsj/P000, {condition}, phase0, {force}, trialSession-{k}, 5s, egocamera{j}_merged.json'
                    csv_file_path = f'data/hsj/P000, {condition}, phase0, {force}, trialSession-{k}, 5s, fsr.csv'
                    video_path = f"data/hsj/P000, {condition}, phase0, {force}, trialSession-{k}, 5s, egocamera{j}_original.mp4"
                    output_dir = f"data/hsj/P000, {condition}, phase0, {force}, trialSession-{k}, 5s, egocamera{j}_original/"
                    json_file_paths.append(json_file_path)
                    csv_file_paths.append(csv_file_path)
                    video_paths.append(video_path)
                    output_dirs.append(output_dir)
    
    for condition in action_label:
        for force in force_label[3:]:
            for k in range(1, 4):
                for j in range(1, 4):
                    json_file_path = f'data/hsj/P000, {condition}, phase1, {force}, trialSession-{k}, 20s, egocamera{j}_merged.json'
                    csv_file_path = f'data/hsj/P000, {condition}, phase1, {force}, trialSession-{k}, 20s, fsr.csv'
                    video_path = f"data/hsj/P000, {condition}, phase1, {force}, trialSession-{k}, 20s, egocamera{j}_original.mp4"
                    output_dir = f"data/hsj/P000, {condition}, phase1, {force}, trialSession-{k}, 20s, egocamera{j}_original/"
                    json_file_paths.append(json_file_path)
                    csv_file_paths.append(csv_file_path)
                    video_paths.append(video_path)
                    output_dirs.append(output_dir)

    for mp4_path, json_path, output_dir in zip(video_paths, json_file_paths, output_dirs):
        crop_frames_from_video(mp4_path, json_path, output_dir)


    # for i in range(0, 5):
    #     for j in range(1, 4):
    #         mp4_path = f'data/P000, CNA, phase0, Positive, trialSession-{i}, 180s, egocamera{j}_original.mp4'
    #         json_path = f'data/P000, CNA, phase0, Positive, trialSession-{i}, 180s, egocamera{j}_merged.json'
    #         output_dir = f'data/P000, CNA, phase0, Positive, trialSession-{i}, 180s, egocamera{j}_original/'

    #         crop_frames_from_video(mp4_path, json_path, output_dir)