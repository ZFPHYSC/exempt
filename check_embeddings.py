#!/usr/bin/env python3
"""
Script to check if embeddings are being stored properly.
"""

import os
import json
import argparse
from pathlib import Path

def check_embeddings_directory(base_dir=None):
    """Check the embeddings directory structure and contents."""
    if base_dir is None:
        # Try to detect the project root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one directory if we're in 'backend'
        if os.path.basename(current_dir) == 'backend':
            base_dir = os.path.dirname(current_dir)
        else:
            base_dir = current_dir
    
    # Check main embeddings directory
    embeddings_dir = os.path.join(base_dir, 'data', 'embeddings')
    print(f"\n=== Checking Embeddings at {embeddings_dir} ===")
    
    if not os.path.exists(embeddings_dir):
        print(f"❌ Embeddings directory doesn't exist: {embeddings_dir}")
        print(f"Creating directory: {embeddings_dir}")
        os.makedirs(embeddings_dir, exist_ok=True)
        return
    
    # Check permissions
    readable = os.access(embeddings_dir, os.R_OK)
    writable = os.access(embeddings_dir, os.W_OK)
    executable = os.access(embeddings_dir, os.X_OK)
    
    print(f"Embeddings directory permissions: Read: {'✅' if readable else '❌'}, "
          f"Write: {'✅' if writable else '❌'}, "
          f"Execute: {'✅' if executable else '❌'}")
    
    # List course directories
    course_dirs = [d for d in os.listdir(embeddings_dir) 
                   if os.path.isdir(os.path.join(embeddings_dir, d))]
    
    if not course_dirs:
        print("❌ No course directories found.")
        return
    
    print(f"✅ Found {len(course_dirs)} course directories: {', '.join(course_dirs)}")
    
    # Check each course directory
    total_docs = 0
    total_vectors = 0
    
    for course_id in course_dirs:
        course_dir = os.path.join(embeddings_dir, course_id)
        doc_files = [f for f in os.listdir(course_dir) 
                     if f.endswith('.json') and os.path.isfile(os.path.join(course_dir, f))]
        
        if not doc_files:
            print(f"❌ No document files found in course: {course_id}")
            continue
        
        print(f"✅ Course {course_id}: Found {len(doc_files)} document files")
        total_docs += len(doc_files)
        
        # Check one document file as a sample
        sample_file = os.path.join(course_dir, doc_files[0])
        try:
            with open(sample_file, 'r') as f:
                vectors = json.load(f)
                
            if not vectors:
                print(f"❌ Empty vectors file: {sample_file}")
                continue
                
            print(f"✅ Sample file {doc_files[0]}: Contains {len(vectors)} vectors")
            
            # Verify vector structure
            sample_vector = vectors[0]
            if 'id' in sample_vector and 'vector' in sample_vector and 'payload' in sample_vector:
                print(f"✅ Vector structure looks good")
                vector_dim = len(sample_vector['vector'])
                print(f"✅ Vector dimension: {vector_dim}")
            else:
                print(f"❌ Invalid vector structure: {list(sample_vector.keys())}")
                
            # Count total vectors
            for doc_file in doc_files:
                file_path = os.path.join(course_dir, doc_file)
                with open(file_path, 'r') as f:
                    file_vectors = json.load(f)
                    total_vectors += len(file_vectors)
                    
        except Exception as e:
            print(f"❌ Error reading vector file {sample_file}: {str(e)}")
    
    # Summary
    print("\n=== Summary ===")
    print(f"Total courses: {len(course_dirs)}")
    print(f"Total document files: {total_docs}")
    print(f"Total vector embeddings: {total_vectors}")
    
    if total_vectors > 0:
        print("\n✅ SUCCESS: Embeddings are being stored correctly!")
    else:
        print("\n❌ ERROR: No vector embeddings found. Check the application logs.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check embeddings storage")
    parser.add_argument("--dir", help="Base directory (project root)", default=None)
    args = parser.parse_args()
    
    check_embeddings_directory(args.dir) 