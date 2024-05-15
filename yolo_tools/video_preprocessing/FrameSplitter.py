import cv2
import os
import numpy as np
from tqdm import tqdm
import argparse


class FrameSplitter:
    def __init__(self, cam_setup='food_cam'):
        if cam_setup == 'food_cam':
            self.camera_matrix = np.array([[1000, 0, 261], [0, 869, 194], [0, 0, 1]])
            self.dist_coeffs = np.array([-0.052, 0.0, 0.000, 0.015, -0.060])
            self.offset_x = 0
            self.offset_y = 0
            self.rows = 9
            self.cols = 6
        else:
            raise ValueError(f'Camera setup is not yet defined in frame splitter: {cam_setup}')
    
    def image_manipulation(self, frame):
        # Define parameters
        new_width = frame.shape[1]
        new_height = frame.shape[0]
        image = self.create_offset_image(frame, new_width, new_height)
        # Create the new image with the specified offset
        return self.undistort_image(image)
    
    def split_frame(self, frame):
        cell_list = []
        x_spacing = frame.shape[1] // self.cols
        y_spacing = frame.shape[0] // self.rows
        
        # Split the frame into cells and save each cell as a separate image
        for y in range(self.offset_y, frame.shape[0], y_spacing):
            for x in range(self.offset_x, frame.shape[1], x_spacing):
                cell = frame[y:y + y_spacing, x:x + x_spacing]
                if cell.shape[0] == y_spacing and cell.shape[1] == x_spacing:
                    cell_list.append(cell)
        
        return cell_list
    
    def write_out_split_cells_as_images(self, cell_list, output_folder, frame_count):
        for cell_num, cell in enumerate(cell_list):
            cell_path = f"{output_folder}/frame_{frame_count}_cell_{cell_num}.jpg"
            cv2.imwrite(cell_path, cell)

    def get_original_video_info(self, cap):
        # Extract properties from the original video
        fps = cap.get(cv2.CAP_PROP_FPS)
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec = int(cap.get(cv2.CAP_PROP_FOURCC))

        # Define new dimensions for the split cells
        new_width = original_width // self.cols
        new_height = original_height // self.rows

        return fps, codec, new_height, new_width

    def start_video_writers(self, video_path, cap, output_folder):
        fps, codec, new_height, new_width = self.get_original_video_info(cap)
        if codec == 0:
            codec = cv2.VideoWriter_fourcc(*'mp4v')  # Fallback to a common codec
        # Initialize video writers for each cell
        writers = []
        for row in range(self.rows):
            for col in range(self.cols):
                video_number = row * self.cols + col # To create a suffix from 01 to self.rows*self.cols
                filename = f"{os.path.splitext(os.path.basename(video_path))[0]}__{video_number:02d}.mp4"
                filepath = os.path.join(output_folder, filename)
                writer = cv2.VideoWriter(filepath, codec, fps, (new_width, new_height))
                writers.append(writer)
        return writers

    def split_video_into_frames(self, video_path, output_folder, max_frames=np.inf):
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        # Read video
        while cap.isOpened():
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret and frame_count < max_frames:
                modified_image = self.image_manipulation(frame)
                cell_list = self.split_frame(modified_image)
                self.write_out_split_cells_as_images(cell_list, output_folder, frame_count)
                frame_count += 1
            else:
                break

        # Release the video capture object
        cap.release()
        cv2.destroyAllWindows()
    
    def split_video_into_videos(self, video_path, output_folder, max_frames=np.inf):
        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Determine the total number of frames in the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Use the smaller of total_frames and max_frames for the progress bar
        frames_to_process = min(total_frames, max_frames)

        writers = self.start_video_writers(video_path, cap, output_folder)
        frame_count = 0

        # Initialize tqdm progress bar
        with tqdm(total=frames_to_process, desc="Splitting Video") as pbar:
            # Read video
            while cap.isOpened():
                # Capture frame-by-frame
                ret, frame = cap.read()
                if ret and frame_count < max_frames:
                    modified_image = self.image_manipulation(frame)
                    cell_list = self.split_frame(modified_image)

                    # Write each cell to its corresponding video
                    for cell, writer in zip(cell_list, writers):
                        writer.write(cell)
                        
                    frame_count += 1
                    pbar.update(1)  # Update the progress bar
                else:
                    break

        # Release the video capture object
        cap.release()
        for writer in writers:
            writer.release()
        cv2.destroyAllWindows()

    def create_offset_image(self, original_image, new_width, new_height):
        # Create a white canvas with the desired size
        new_image = 255 * np.ones((new_height, new_width, 3), dtype=np.uint8)

        # Calculate the position to paste the original image with the specified offset
        paste_position_x = self.offset_x
        paste_position_y = self.offset_y

        # Paste the original image onto the white canvas
        new_image[paste_position_y:paste_position_y + original_image.shape[0],
                  paste_position_x:paste_position_x + original_image.shape[1]] = original_image
        return new_image
    
    def undistort_image(self, img):
        # Undistort the image
        dst = cv2.undistort(img, self.camera_matrix, self.dist_coeffs, None, self.camera_matrix)
        return dst


def main():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Split video into frames or smaller videos.')
    parser.add_argument('--video_path', type=str, required=True, help='Path to the original video.')
    parser.add_argument('--output_folder', type=str, required=True, help='Output directory path.')
    parser.add_argument('--output_type', type=str, choices=['frames', 'videos'], required=True, help='Output as frames or videos.')
    parser.add_argument('--max_frames', type=int, default=np.inf, help='Maximum number of frames to process.')
    
    # Parse the arguments
    args = parser.parse_args()

    # Initialize FrameSplitter
    fs = FrameSplitter()

    # Check the output type and call the appropriate method
    if args.output_type == 'frames':
        fs.split_video_into_frames(args.video_path, args.output_folder, args.max_frames)
    elif args.output_type == 'videos':
        fs.split_video_into_videos(args.video_path, args.output_folder, args.max_frames)


if __name__ == '__main__':
    main()
