import os
import cv2
import random

def choose_random_file(directory):
    all_files = os.listdir(directory)
    files = [f for f in all_files if os.path.isfile(os.path.join(directory, f))]

    if not files:
        raise FileNotFoundError("No files found in the directory")

    random_file = random.choice(files)
    return os.path.join(directory, random_file)

to_compare_directory = "SOCOFing/Altered/Altered-Hard/"

try:
    random_file = choose_random_file(to_compare_directory)
    print(f"Randomly selected file: {random_file}")
except FileNotFoundError as e:
    print(e)
    exit()

sample = cv2.imread(random_file)
if sample is None:
    print("Error: Could not load the sample image.")
    exit()

filename = None
image = None
kp1, kp2, mp = None, None, None
best_score = 0
real_directory = "SOCOFing/Real/"

sift = cv2.SIFT_create()
keypoints_1, descriptors_1 = sift.detectAndCompute(sample, None)

for file in os.listdir(real_directory):
    fingerprint_image = cv2.imread(os.path.join(real_directory, file))
    if fingerprint_image is None:
        print(f"Error: Could not load fingerprint image {file}. Skipping.")
        continue

    keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_image, None)

    matches = cv2.FlannBasedMatcher(
        {'algorithm': 1, 'trees': 10},
        {}
    ).knnMatch(descriptors_1, descriptors_2, k=2)

    match_points = []

    for p, q in matches:
        if p.distance < 0.1 * q.distance:
            match_points.append(p)
    
    keypoints_count = min(len(keypoints_1), len(keypoints_2))

    if keypoints_count > 0:
        score = len(match_points) / keypoints_count * 100
        if score > best_score:
            best_score = score
            filename = file
            image = fingerprint_image
            kp2 = keypoints_2
            mp = match_points

print("Best Match: " + (filename if filename else "No Match Found"))
print("Score: " + str(best_score))

if image is not None and kp1 is not None and kp2 is not None and match_points is not None:
    result = cv2.drawMatches(sample, kp1, image, kp2, mp, None)
    result = cv2.resize(result, None, fx=4, fy=4)
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No matches to display.")
