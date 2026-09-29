from subprocess import getstatusoutput

PRG = "HoverFast"


def test_general_help_works() -> None:
    """-h option prints help page"""
    rv, out = getstatusoutput(f"{PRG} -h")
    assert rv == 0
    assert out.lower().startswith("usage:")


def test_infer_wsi_help_works() -> None:
    """-h option prints help page"""
    rv, out = getstatusoutput(f"{PRG} infer_wsi -h")
    assert rv == 0
    assert out.lower().startswith("usage:")


def test_infer_roi_help_works() -> None:
    """-h option prints help page"""
    rv, out = getstatusoutput(f"{PRG} infer_roi -h")
    assert rv == 0
    assert out.lower().startswith("usage:")


def test_train_help_works() -> None:
    """-h option prints help page"""
    rv, out = getstatusoutput(f"{PRG} train -h")
    assert rv == 0
    assert out.lower().startswith("usage:")


def test_build_help_works() -> None:
    """-h option prints help page for the TensorRT build sub-command"""
    rv, out = getstatusoutput(f"{PRG} build -h")
    assert rv == 0
    assert out.lower().startswith("usage:")
    assert "--engine_path" in out


def test_versioning() -> None:
    """-h option prints help page"""
    rv, out = getstatusoutput(f"{PRG} --version")
    assert rv == 0
    assert out.lower().startswith("hoverfast")
