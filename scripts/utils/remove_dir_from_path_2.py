"""
Vynecha retezce typu BK01_100fps a BK01_30fps z cesty. Pred spustenim zkontrolovat obsah adresaru!
"""

import os
import glob
import shutil

# Find all directories ending with '_100fps' or '_30fps'
fps_dirs = glob.glob('/mnt/ERC/ERC/Mereni_Babice/**/*_100fps', recursive=True) + glob.glob('/mnt/ERC/ERC/Mereni_Babice/**/*_30fps', recursive=True)

# Iterate over found directories
for fps_dir in fps_dirs:
    if os.path.isdir(fps_dir):
        parent_dir = os.path.dirname(fps_dir)
        print(f"Moving contents of directory: {fps_dir}")

        # Move each item in the directory to the parent directory
        for item in os.listdir(fps_dir):
            src_item = os.path.join(fps_dir, item)
            dst_item = os.path.join(parent_dir, item)
            
            # Move the item (file or directory)
            shutil.move(src_item, dst_item)
            print(f"  Moved {src_item} to {dst_item}")

        # Check if the fps_dir is empty and remove it
        if not os.listdir(fps_dir):
            os.rmdir(fps_dir)
            print(f"  Removed empty directory: {fps_dir}")
