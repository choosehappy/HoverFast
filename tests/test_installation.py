from subprocess import getstatusoutput

PRG = 'HoverFast'

def test_general_help_works() -> None:
    """ -h option prints help page """
    rv, out = getstatusoutput(f'{PRG} -h')
    assert rv == 0
    assert out.lower().startswith('usage:')

def test_infer_wsi_help_works() -> None:
    """ -h option prints help page """
    rv, out = getstatusoutput(f'{PRG} infer_wsi -h')
    assert rv == 0
    assert out.lower().startswith('usage:')

def test_infer_roi_help_works() -> None:
    """ -h option prints help page """
    rv, out = getstatusoutput(f'{PRG} infer_roi -h')
    assert rv == 0
    assert out.lower().startswith('usage:')

def test_train_help_works() -> None:
    """ -h option prints help page """
    rv, out = getstatusoutput(f'{PRG} infer_roi -h')
    assert rv == 0
    assert out.lower().startswith('usage:')

def test_versioning() -> None:
    """ -h option prints help page """
    rv, out = getstatusoutput(f'{PRG} --version')
    assert rv == 0
    assert out.lower().startswith('hoverfast')
