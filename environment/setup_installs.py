# Delete assets.zip and /content/assets/
import shutil, gdown, os

if os.path.exists("assets"):
    shutil.rmtree("assets")
if os.path.exists("assets.zip"):
    os.remove("assets.zip")

# Redownload from Drive
data_path = "assets.zip"
print("Downloading assets.zip...")
url = (
    "https://drive.google.com/file/d/1F2MJQ5enUPVtyi3s410PUuv8LiWr8qCz/view?usp=sharing"
)
gdown.download(url, output=data_path, fuzzy=True)


# Delete attacks.zip and /content/attacks/
if os.path.exists("attacks"):
    shutil.rmtree("attacks")
if os.path.exists("attacks.zip"):
    os.remove("attacks.zip")

# Redownload from Drive
data_path = "attacks.zip"
print("Downloading attacks.zip...")
url = (
    "https://drive.google.com/file/d/1LAOL8sYCUfsCk3TEA3vvyJCLSl0EdwYB/view?usp=sharing"
)
gdown.download(url, output=data_path, fuzzy=True)
