import os
import subprocess

def create_video_from_images(image_dir, output_file):
    """
    Creates a video from images in a directory, using the order they are listed.

    Args:
        image_dir (str): Path to the directory containing the images.
        output_file (str): Path to save the output video.
    """

    # 1. Get a list of the image files (no sorting)
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]

    ffmpeg_command = [
        "ffmpeg",
        "-framerate", "30", #set frame rate
        "-i", os.path.join(image_dir, "%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", #scale to even height.
        output_file
    ]

    #Rename the files to a sequence.
    renamed_files = []
    original_files = []

    for i, file in enumerate(image_files):
        original_files.append(file)
        new_name = f"{i+1:04d}.png"
        os.rename(os.path.join(image_dir, file), os.path.join(image_dir, new_name))
        renamed_files.append(new_name)

    subprocess.run(ffmpeg_command, check=True)

    #Rename the files back.
    for original, renamed in zip(original_files, renamed_files):
        os.rename(os.path.join(image_dir, renamed), os.path.join(image_dir, original))


# Example usage
image_directory = "/media/GLAB/AI_trainData/uLytics_Lena/images/val"  # Replace with the actual path
output_video = "/home/geuba03p/all_models_comparison/test_vid.mp4"  # Replace with the desired output path

create_video_from_images(image_directory, output_video)