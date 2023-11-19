# import os
# import librosa

# # Set directories
# ogg_dir = "data_narrative/ogg/open_images_train_v6_localized_narratives-00000-of-00010"
# image_features_dir = "data_narrative/image_features/open_images_train_v6_localized_narratives-00000-of-00010"
# txt_file = "data_narrative/data_list/open_images_train_v6_localized_narratives-00000-of-00010.txt"

# # Loop through ogg files
# for filename in os.listdir(ogg_dir):
#     if filename.endswith(".ogg"):
#         # Extract duration
#         full_path = os.path.join(ogg_dir, filename)
#         duration = librosa.get_duration(filename=full_path)
        
#         # Check duration and remove file and npy file if necessary
#         if duration >= 30:
#             # os.remove(full_path)
#             # npy_path = os.path.join(image_features_dir, f"{filename[:-4]}.npy")
#             # if os.path.exists(npy_path):
#             #     os.remove(npy_path)
#             # Remove filename from txt file
#             with open(txt_file, 'r') as f:
#                 lines = f.readlines()
#             with open(txt_file, 'w') as f:
#                 for line in lines:
#                     if filename not in line:
#                         f.write(line)
# filename = "path/to/your/file.txt" # Replace with your filename

# with open(filename, "r") as f:
#     lines = f.readlines()

# new_lines = []
# for line in lines:
#     if len(line) <= 120:
#         new_lines.append(line)

# with open(filename, "w") as f:
#     f.writelines(new_lines)
import os
import librosa
ogg_dir = "data_narrative/ogg/open_images_train_v6_localized_narratives-00000-of-00010"
filename = "data_narrative/data_list/open_images_train_v6_localized_narratives-00000-of-00010.txt"


with open(filename, "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    image_id, caption, url = line.split("\t")
    image_id=image_id.split('/')[-1]
    # Check caption length and ogg file duration
    if len(caption) <= 100000:
        ogg_file = os.path.join(ogg_dir, f"{image_id}.ogg")
        if os.path.exists(ogg_file):
            duration = librosa.get_duration(filename=ogg_file)
            if duration < 30:
                new_lines.append(line)
                
# Write filtered lines to file
with open(filename, "w") as f:
    f.writelines(new_lines)
