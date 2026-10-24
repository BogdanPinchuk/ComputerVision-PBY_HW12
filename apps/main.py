import cv2
import numpy as np
from pathlib import Path
from packaging.requirements import Requirement
from importlib.metadata import PackageNotFoundError, version


def load_image(file_name: str) -> np.ndarray:
    """
    Load image by file path
    :param file_name: file name
    :return: image
    """
    file_path = Path(file_name)

    # Load an image (you can freely choose any image you like)
    if file_path.exists():
        img_bgr = cv2.imread(str(file_path))
    else:
        raise FileNotFoundError(f"File '{file_name}' doesn't exist!")

    if img_bgr is None:
        raise ValueError("Incorrect input data! Could not decode image: '{file_name}'")

    # Convert it to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    return img_rgb

def is_package_satisfied(req_str: str) -> bool:
  try:
    req = Requirement(req_str)
    installed_ver = version(req.name)
    if req.specifier:
      return installed_ver in req.specifier
    return True
  except PackageNotFoundError:
    return False