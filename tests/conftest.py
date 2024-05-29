import hashlib
import pathlib
import shutil
import urllib.request
import pytest

# openslide aperio test images
IMAGES_BASE_URL = "https://data.cytomine.coop/open/openslide/aperio-svs/"

def md5(fn):
    m = hashlib.md5()
    with open(fn, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            m.update(chunk)
    return m.hexdigest()

@pytest.fixture(scope='session')
def svs_small():
    """download the small aperio test image svs"""
    small_image = "CMU-1.svs"
    small_image_md5 = "751b0b86a3c5ff4dfc8567cf24daaa85"
    data_dir = pathlib.Path(__file__).parent / "data"

    data_dir.mkdir(parents=True, exist_ok=True)
    img_fn = data_dir / small_image

    if not img_fn.is_file():
        print(f"Downloading {small_image} to {img_fn}")
        # download svs from openslide test images
        url = IMAGES_BASE_URL + small_image
        with urllib.request.urlopen(url) as response, open(img_fn, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)

    if md5(img_fn) != small_image_md5:  # pragma: no cover
        print(f"MD5 mismatch for {small_image}, deleting file.")
        shutil.rmtree(img_fn)
        pytest.fail("incorrect md5")
    else:
        print(f"Downloaded and verified {small_image}")
        yield img_fn.absolute()
