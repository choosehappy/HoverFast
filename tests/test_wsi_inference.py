import os
import shutil
import textwrap
import hashlib
from conftest import *
import pytest
import openslide
import cv2
import numpy as np
from subprocess import getstatusoutput, getoutput

PRG = 'HoverFast'
MODEL_PATH = 'hoverfast_crosstissue_best_model.pth'
ABS_PATH_MODEL = os.path.abspath(MODEL_PATH)
SLIDE_PATH = 'tests/data/'
ABS_SLIDE_PATH = os.path.abspath(SLIDE_PATH)

# --- small helpers ---
def _filenames_in(pth): return set(x.name for x in pth.glob('*'))

@pytest.fixture(scope='module')
def single_svs_dir(tmp_path_factory, svs_small):
    pth = tmp_path_factory.mktemp('hoverfast_wsi_test_single')
    shutil.copy(svs_small, pth)
    yield pth

def test_infer_wsi(tmp_path, single_svs_dir):
    print(f"Running infer_wsi")
    svs_dir = os.fspath(single_svs_dir)
    rv, out = getstatusoutput(f"{PRG} infer_wsi {svs_dir}/*.svs -m {ABS_PATH_MODEL} -b {SLIDE_PATH} -l 20.0 -o {tmp_path} -n 4")
    print(f"Return value: {rv}")
    print(f"Output: {out}")
    assert rv == 0

    # Check for log file
    log_files = list(tmp_path.rglob('*.log'))
    print(f"Log files found: {log_files}")
    assert len(log_files) > 0, "No log file found"

    # Check for .json.gz file
    json_gz_files = list(tmp_path.rglob('*.json.gz'))
    print(f".json.gz files found: {json_gz_files}")
    assert len(json_gz_files) > 0, "No .json.gz file found"
