import os

folder_path = "/home/shannon/local/Source/Python/bm_study/images/masks"

for filename in os.listdir(folder_path):
    if filename.endswith(".tif"):
        new_filename = os.path.splitext(filename)[0] + ".tiff"
        old_filepath = os.path.join(folder_path, filename)
        new_filepath = os.path.join(folder_path, new_filename)
        os.rename(old_filepath, new_filepath)
        print(f"Renamed {filename} to {new_filename}")

print("Renaming complete!")
