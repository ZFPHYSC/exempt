#!/usr/bin/env python3
"""
Simple script to manually migrate embeddings to named folders.
This doesn't require database access.
"""

import os
import shutil
import json
from pathlib import Path

# Data directory where embeddings are stored
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "embeddings")

def migrate_embeddings():
    """Manually migrate embeddings to use course name-based folders"""
    print(f"Starting manual embedding migration to course name-based folders")
    print(f"Looking in: {DATA_DIR}")
    
    if not os.path.exists(DATA_DIR):
        print(f"Error: Embeddings directory doesn't exist: {DATA_DIR}")
        return
    
    # Find all course ID directories
    migrated = 0
    errors = 0
    
    for dir_name in os.listdir(DATA_DIR):
        dir_path = os.path.join(DATA_DIR, dir_name)
        
        # Skip if not a directory or already in named format
        if not os.path.isdir(dir_path) or '_' in dir_name:
            continue
        
        # Try to find course name from any file
        course_name = None
        for file_name in os.listdir(dir_path):
            if file_name.endswith('.json'):
                try:
                    with open(os.path.join(dir_path, file_name), 'r') as f:
                        data = json.load(f)
                        if data and isinstance(data, list) and len(data) > 0:
                            # Try to extract course name from payload metadata
                            vector = data[0]
                            if 'payload' in vector and 'metadata' in vector['payload']:
                                metadata = vector['payload']['metadata']
                                if 'filename' in metadata:
                                    # Use filename as course name
                                    course_name = metadata['filename'].replace('.', '_')
                                    break
                except Exception as e:
                    print(f"Error reading file {file_name}: {e}")
        
        if not course_name:
            print(f"Could not determine course name for {dir_name}, skipping")
            continue
            
        # Create new directory name
        new_dir_name = f"{dir_name}_{course_name}"
        new_dir_path = os.path.join(DATA_DIR, new_dir_name)
        
        try:
            # Create new directory
            os.makedirs(new_dir_path, exist_ok=True)
            
            # Move all files
            files_moved = 0
            for file_name in os.listdir(dir_path):
                if file_name.endswith('.json'):
                    old_file_path = os.path.join(dir_path, file_name)
                    new_file_path = os.path.join(new_dir_path, file_name)
                    # Copy the file
                    shutil.copy2(old_file_path, new_file_path)
                    files_moved += 1
            
            print(f"Migrated {files_moved} files from {dir_name} to {new_dir_name}")
            migrated += 1
            
        except Exception as e:
            print(f"Error migrating course {dir_name}: {e}")
            errors += 1
    
    print(f"\nMigration completed:")
    print(f"  - Courses migrated: {migrated}")
    print(f"  - Errors: {errors}")
    
    if migrated > 0:
        print("\nMigration successful! Embeddings are now organized by course name.")
        print("Note: Original folders were kept as backup. Once you verify everything works,")
        print("you can manually delete the original folders (those without underscores).")
    else:
        print("\nNo changes were made during migration.")

if __name__ == "__main__":
    migrate_embeddings() 