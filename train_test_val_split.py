import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
src_dir = "dataset"
dst_dir = "dataset_split"

# Create destination folders
splits = ["train", "test", "val"]
for split in splits:
    for obj in os.listdir(src_dir):
        os.makedirs(os.path.join(dst_dir, split, obj), exist_ok=True)

# Loop over each object folder
for obj in os.listdir(src_dir):
    obj_path = os.path.join(src_dir, obj)
    if not os.path.isdir(obj_path):
        continue
    
    images = os.listdir(obj_path)
    
    # Shuffle + split into train(20), test(10), val(10)
    train_imgs, temp_imgs = train_test_split(images, train_size=20, random_state=42)
    test_imgs, val_imgs = train_test_split(temp_imgs, test_size=10, random_state=42)  # 10 test, 10 val
    
    # Copy files
    for img in train_imgs:
        shutil.copy(os.path.join(obj_path, img), os.path.join(dst_dir, "train", obj))
    for img in test_imgs:
        shutil.copy(os.path.join(obj_path, img), os.path.join(dst_dir, "test", obj))
    for img in val_imgs:
        shutil.copy(os.path.join(obj_path, img), os.path.join(dst_dir, "val", obj))

print("✅ Dataset split into train=20, test=10, val=10 for each object!")
