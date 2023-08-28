#NOT Done!!

# import os
# import numpy as np
# from PIL import Image
# from skimage.transform import resize
#
# def load_jpg_data(data_dir):
#     jpg_images = []
#     for filename in os.listdir(data_dir):
#         if filename.endswith(".png"):
#             filepath = os.path.join(data_dir, filename)
#             img = Image.open(filepath)
#             jpg_images.append(np.array(img))
#     return np.array(jpg_images)
#
# data_dir = "./img/"
# jpg_images = load_jpg_data(data_dir)
#
# # Define or load triplets
# triplets = [
#     (jpg_images[0], jpg_images[1], jpg_images[2]),
#     #(jpg_images[1], jpg_images[5], jpg_images[2]),
#     #(jpg_images[6], jpg_images[7], jpg_images[8])
# ]
#
# def preprocess_triplets(triplets):
#     preprocessed_triplets = []
#     for anchor, positive, negative in triplets:
#         anchor = resize(anchor, (224, 224))
#         positive = resize(positive, (224, 224))
#         negative = resize(negative, (224, 224))
#
#         preprocessed_triplets.append((anchor, positive, negative))
#     return preprocessed_triplets
#
# preprocessed_triplets = preprocess_triplets(triplets)
#
# # Save preprocessed triplets as images
# output_dir = "./preprocessed_triplets/"
# os.makedirs(output_dir, exist_ok=True)
#
# for idx, (anchor, positive, negative) in enumerate(preprocessed_triplets):
#     anchor_path = os.path.join(output_dir, f"anchor_{idx}.png")
#     positive_path = os.path.join(output_dir, f"positive_{idx}.png")
#     negative_path = os.path.join(output_dir, f"negative_{idx}.png")
#
#     anchor_img = Image.fromarray(anchor.astype(np.uint8))
#     positive_img = Image.fromarray(positive.astype(np.uint8))
#     negative_img = Image.fromarray(negative.astype(np.uint8))
#
#     anchor_img.save(anchor_path)
#     positive_img.save(positive_path)
#     negative_img.save(negative_path)
