import os
from pathlib import Path

def keep_recent_files(folder_path, max_files=50):
    # Get all files in the folder
    files = [f for f in Path(folder_path).iterdir() if f.is_file()]
    
    # Sort files by modification time (most recent first)
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    
    # Remove older files if there are more than max_files
    for file in files[max_files:]:
        os.remove(file)
        print(f"Deleted: {file}")

# List of folder paths
folders = [
    '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/1_blue',
    '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/1_green',
    '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/1_red',
    '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/1_yellow',

        '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/2_blue',
    '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/2_green',
    '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/2_red',
    '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/2_yellow',

        '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/3_blue',
    '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/3_green',
    '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/3_red',
    '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/3_yellow',

        '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/4_blue',
    '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/4_green',
    '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/4_red',
    '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/4_yellow',

        '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/5_blue',
    '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/5_green',
    '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/5_red',
    '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/5_yellow',

        '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/6_blue',
    '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/6_green',
    '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/6_red',
    '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/6_yellow',

        '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/7_blue',
    '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/7_green',
    '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/7_red',
    '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/7_yellow',

        '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/8_blue',
    '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/8_green',
    '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/8_red',
    '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/8_yellow',

            '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/9_blue',
    '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/9_green',
    '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/9_red',
    '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/9_yellow',

    '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/reverse_blue',
    '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/reverse_green',
    '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/reverse_red',
    '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/reverse_yellow',

    '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/skip_blue',
    '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/skip_green',
    '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/skip_red',
    '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/skip_yellow',

    '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/draw2_blue',
    '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/draw2_green',
    '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/draw2_red',
    '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/draw2_yellow',

    '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/wild',
    '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/wild',
    '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/wild',
    '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/wild',

    '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/wild_draw4',
    '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/wild_draw4',
    '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/wild_draw4',
    '/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/wild_draw4',
]

# Apply the function to each folder
for folder_path in folders:
    print(f"Processing folder: {folder_path}")
    keep_recent_files(folder_path)