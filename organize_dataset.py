import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split

meta = pd.read_csv(r"dataset\HAM10000\HAM10000_metadata.csv")

src = r"dataset/HAM10000/images_all"
train_dst = r"dataset/HAM10000/images1"   # ← training goes here
test_dst  = r"dataset/HAM10000/test1"     # ← testing goes here

# stratified split
train_df, test_df = train_test_split(
    meta,
    test_size=0.2,
    stratify=meta["dx"],
    random_state=42
)

def copy_set(df, base_path):
    for _, r in df.iterrows():
        cls = r["dx"]
        img = r["image_id"] + ".jpg"

        dst_dir = os.path.join(base_path, cls)
        os.makedirs(dst_dir, exist_ok=True)

        shutil.copy(
            os.path.join(src, img),
            os.path.join(dst_dir, img)
        )

# copy training → images/
copy_set(train_df, train_dst)

# copy testing → test/
copy_set(test_df, test_dst)

print("Done — training images in 'images', testing images in 'test'")
