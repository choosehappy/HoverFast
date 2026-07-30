import torch
import torch_tensorrt
import modelopt.torch.quantization as mtq
from . import utils_wsi 

from torch.utils.data import DataLoader
from tqdm import tqdm

from modelopt.torch.quantization.utils import export_torch_mode

#conda install -c nvidia cuda-nvcc=13.2

# -----------------------------------------------------------------------------
# Calibration
# -----------------------------------------------------------------------------

def make_calibration_loop(loader, device):
    """
    Returns the callback expected by ModelOpt.
    """

    def calibration_loop(model):
        model.eval()

        with torch.inference_mode():
            for imgs, _ in tqdm(loader, desc="Calibration", leave=False):

                imgs = (
                    imgs.to(
                        device,
                        memory_format=torch.channels_last,
                        non_blocking=True,
                    )
                    .half()
                    .div_(255.0)
                )

                model(imgs)

    return calibration_loop


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------

def build_calibration_loader(
    coords,
    slide_data,
    batch_size,
    n_workers,
):

    dataset = utils_wsi.WSIPatchDataset(coords, slide_data)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=n_workers,
        persistent_workers=n_workers > 0,
        prefetch_factor=4 if n_workers > 0 else None,
    )


# -----------------------------------------------------------------------------
# Quantization
# -----------------------------------------------------------------------------

def quantize_model(
    model,
    loader,
    device,
    precision="int8",
):
    """
    Quantize the model using ModelOpt.

    precision:
        "int8"
        "fp8"
    """

    model = model.eval().half()

    calibration_loop = make_calibration_loop(loader, device)

    if precision == "int8":
        quant_cfg = mtq.INT8_DEFAULT_CFG

    elif precision == "fp8":
        quant_cfg = mtq.FP8_DEFAULT_CFG

    else:
        raise ValueError(f"Unknown precision '{precision}'")

    model = mtq.quantize(
        model,
        quant_cfg,
        forward_loop=calibration_loop,
    )

    print(type(model))
    print(model.down_path[0].conv_3.block[0])

    return model


# -----------------------------------------------------------------------------
# TensorRT
# -----------------------------------------------------------------------------

def compile_tensorrt_engine(
    model,
    precision="fp16",
):

    batch = torch.export.Dim(
        "batch",
        min=1,
        max=7,
    )

    example_input = torch.randn(
        7,
        3,
        1024,
        1024,
        device="cuda",
        dtype=torch.float16,
    )
    with export_torch_mode():
    
        exp_program = torch.export.export(
            model,
            (example_input,),
            dynamic_shapes={
                "x": {0: batch},
            },
        )

    enabled_precisions = {torch.float16}

    if precision == "int8":
        enabled_precisions.add(torch.int8)

    elif precision == "fp8":
        enabled_precisions.add(torch.float8_e4m3fn)

    trt_model = torch_tensorrt.dynamo.compile(
        exp_program,
        inputs=[
            torch_tensorrt.Input(
                min_shape=(1, 3, 1024, 1024),
                opt_shape=(7, 3, 1024, 1024),
                max_shape=(7, 3, 1024, 1024),
                dtype=torch.half,
            )
        ],
        enabled_precisions=enabled_precisions,
        optimization_level=5,
        workspace_size=8 << 30,
        use_python_runtime=False,
    )

    torch_tensorrt.save(trt_model, "unet_trt_int8.ts", inputs=[example_input], output_format="torchscript")#, dynamic_shapes=dynamic_shapes)
    return trt_model


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def calibrate_model(
    sname,
    sformat,
    fpath,
    mask_dir,
    outdir,
    mag,
    batch_to_gpu,
    region_size,
    model,
    device,
    n_process,
    poly_simplify_tolerance,
    threshold,
    stain,
    logger,
    db_output_fname,
    precision="int8",
):

    n_post_proc = max(1, n_process // 2)
    n_loader = max(1, n_process - n_post_proc)

    kernel_size = 256

    slide_data = utils_wsi.get_slide(
        sname,
        sformat,
        fpath,
        mag,
        kernel_size,
        region_size,
        threshold,
        outdir,
        poly_simplify_tolerance,
        logger,
    )

    coords = utils_wsi.find_regions(mask_dir, slide_data)

    print(f"|- Calibration using {len(coords)} patches.")

    loader = build_calibration_loader(
        coords=coords,
        slide_data=slide_data,
        batch_size=batch_to_gpu,
        n_workers=n_loader,
    )

    model = quantize_model(
        model=model,
        loader=loader,
        device=device,
        precision=precision,
    )

    trt_model = compile_tensorrt_engine(
        model,
        precision=precision,
    )


    return trt_model