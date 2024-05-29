import os
import shutil
import textwrap
import pytest
import openslide
import cv2
from subprocess import getstatusoutput, getoutput


PRG = 'HoverFast'
DATASET_PATH = 'tests/data/'
ABS_DATASET_PATH = os.path.abspath(DATASET_PATH)

def _filenames_in(pth): return set(x.name for x in pth.glob('*'))

@pytest.fixture(scope='module')
def single_svs_dir(tmp_path_factory):
    pth = tmp_path_factory.mktemp('hoverfast_test_single')
    yield pth


def test_train(tmp_path):
    rv, out = getstatusoutput(f"{PRG} train testing_dataset -p {ABS_DATASET_PATH} -b 1 -e 1 -o {tmp_path}")
    assert rv == 0

    # Check for directory containing 'hoverfast'
    hoverfast_dirs = [d for d in tmp_path.iterdir() if d.is_dir() and 'hoverfast' in d.name]
    assert len(hoverfast_dirs) > 0, "No directory containing 'hoverfast' found"

    # Check for file with .pth extension
    pth_files = list(tmp_path.rglob('*.pth'))
    assert len(pth_files) > 0, "No file with .pth extension found"
