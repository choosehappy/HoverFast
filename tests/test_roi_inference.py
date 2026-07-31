import os
from subprocess import getstatusoutput

import pytest

PRG = 'HoverFast'
MODEL_PATH = 'hoverfast_crosstissue_best_model.pth'
ABS_PATH_MODEL = os.path.abspath(MODEL_PATH)
TILE_PATH = 'tests/data/'
ABS_TILE_PATH = os.path.abspath(TILE_PATH)

def _filenames_in(pth): return {x.name for x in pth.glob('*')}

@pytest.fixture(scope='module')
def single_roi_dir(tmp_path_factory):
    pth = tmp_path_factory.mktemp('hoverfast_roi_test_single')
    yield pth


def test_infer_roi_he(tmp_path):
    rv, out = getstatusoutput(f"{PRG} infer_roi {ABS_TILE_PATH}/*.png -m {ABS_PATH_MODEL} -o {tmp_path}")
    assert rv == 0
    assert _filenames_in(tmp_path) == _filenames_in(tmp_path).union(["json", "label_mask", "overlay"])

def test_infer_roi_ihc(tmp_path):
    rv, out = getstatusoutput(f"{PRG} infer_roi {ABS_TILE_PATH}/*.png -m {ABS_PATH_MODEL} -o {tmp_path} -st ihc_dab")
    assert rv == 0
    assert _filenames_in(tmp_path) == _filenames_in(tmp_path).union(["json", "label_mask", "overlay"])

