import os
import zipfile
from pathlib import Path

import requests

data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

# if the image folder doesn't exist, download it and prepare it"
if not image_path.is_dir():
    print(f"{image_path} does not exist, downloading and preparing it.")
else:
    print(f"{image_path} already exists, skipping download.")
    image_path.mkdir(parents=True, exist_ok=True)


# ✅ Ensure the data/ directory exists
data_path.mkdir(parents=True, exist_ok=True)

# download the dataset
with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
    url = "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip"
    request = requests.get(url)
    f.write(request.content)

# extract the dataset unzipping
with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
    zip_ref.extractall(image_path)
# remove the zip file
os.remove(data_path / "pizza_steak_sushi.zip")
