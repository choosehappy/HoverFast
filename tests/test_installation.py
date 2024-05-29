import os
import platform
import re
from subprocess import getstatusoutput, getoutput

PRG = 'HoverFast'

def test_general_help_works() -> None:
    """ -h option prints help page """
    rv, out = getstatusoutput('{} -h'.format(PRG))
    assert rv == 0
    assert out.lower().startswith('usage:')

def test_infer_wsi_help_works() -> None:
    """ -h option prints help page """
    rv, out = getstatusoutput('{} infer_wsi -h'.format(PRG))
    assert rv == 0
    assert out.lower().startswith('usage:')

def test_infer_roi_help_works() -> None:
    """ -h option prints help page """
    rv, out = getstatusoutput('{} infer_roi -h'.format(PRG))
    assert rv == 0
    assert out.lower().startswith('usage:')

def test_train_help_works() -> None:
    """ -h option prints help page """
    rv, out = getstatusoutput('{} infer_roi -h'.format(PRG))
    assert rv == 0
    assert out.lower().startswith('usage:')

def test_versioning() -> None:
    """ -h option prints help page """
    rv, out = getstatusoutput('{} --version'.format(PRG))
    assert rv == 0
    assert out.lower().startswith('hoverfast')