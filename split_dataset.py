import os
import shutil
from sklearn.model_selection import train_test_split

source = "dataset_raw"
dest = "dataset"
classes = ["NE", "EP", "EH", "EA"]

for cls in classes:
    class_path = os.path.join(source, cls)
    files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg','.jpeg','.png'))]

    train, temp = train_test_split(files, test_size=0.30, random_state=42)
    val, test = train_test_split(temp, test_size=0.50, random_state=42)

    for split_name, split_files in {
        "train": train,
        "val": val,
        "test": test
    }.items():

        folder = os.path.join(dest, split_name, cls)
        os.makedirs(folder, exist_ok=True)

        for file in split_files:
            shutil.copy(
                os.path.join(class_path, file),
                os.path.join(folder, file)
            )

print("Dataset split completed!")