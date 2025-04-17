import os
import urllib.request

# Create models folder if not exists
os.makedirs("models", exist_ok=True)

# Fixed working URLs
prototxt_url = "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/models/gender_deploy.prototxt"
caffemodel_url = "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/models/gender_net.caffemodel"

# File paths
prototxt_path = "models/gender_deploy.prototxt"
caffemodel_path = "models/gender_net.caffemodel"

# Download function
def download(url, path):
    if not os.path.exists(path):
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, path)
        print(f"Saved to {path}")
    else:
        print(f"{path} already exists. Skipping.")

# Download both files
download(prototxt_url, prototxt_path)
download(caffemodel_url, caffemodel_path)
