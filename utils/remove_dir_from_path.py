"""
Z cesty ../zpracovana_data/exports_xsight/... vynechat exports_xsight
"""

import glob
import os
import shutil

#dry_run = False
dry_run = True

# Find all instances of 'exports_xsight' directories
exports_xsight_dirs = glob.glob('/mnt/ERC/ERC/Mereni_Babice/**/zpracovana_data/exports_xsight', recursive=True)
print(exports_xsight_dirs)  # zkontroluje soubory, ktere byly nalezeny

for src_dir in exports_xsight_dirs:
    dst_dir = os.path.dirname(src_dir)  # Parent directory of 'exports_xsight'
    
    # Move each file and subdirectory in the exports_xsight directory to its parent directory
    for item in os.listdir(src_dir):
        src_item = os.path.join(src_dir, item)
        dst_item = os.path.join(dst_dir, item)
        
        # Move the item (file or directory)
        # 
        if dry_run:
            print(src_item,"=>", dst_item)
        else:
            shutil.move(src_item, dst_item)
            
    # Check if the exports_xsight directory is empty
    if not os.listdir(src_dir):
        # Remove the exports_xsight directory after moving its contents
        if dry_run:
            print(f"Mazu {src_dir}")
        else:
            shutil.rmtree(src_dir)
        
    else:
        print(f"Adresar {src_dir} neni prazdny")