import click
from Expresssion2RGB import Expression2Img
import cv2
from utils import PathHandler, VisiumPaths
from pathlib import Path
import os

@click.command()
@click.option("--visium_dir", type=Path)
@click.option("--out_dir", type=Path)
@click.option("--k", type=int, default=256)
@click.option("--alpha", type=float, default=0.5)
def cli(visium_dir: Path, out_dir:Path, k: int, alpha: float):
    visium_dir = PathHandler.validatePath(str(visium_dir))
    assert PathHandler.checkDir(visium_dir), f"{visium_dir} dont exist"
    h5_path = f"{visium_dir}{VisiumPaths.h5_path}"
    spatial_path = f"{visium_dir}{VisiumPaths.spatial_path}"
    scale_factor_path = f"{visium_dir}{VisiumPaths.scale_factor_path}"
    assert PathHandler.checkFile(h5_path), f"{h5_path} dont exist" 
    assert PathHandler.checkFile(spatial_path), f"{spatial_path} dont exist" 
    assert PathHandler.checkFile(scale_factor_path), f"{scale_factor_path} dont exist"
    out_dir = PathHandler.validatePath(str(out_dir))
    assert PathHandler.checkDir(out_dir), f"{out_dir} dont exist"

    E2I = Expression2Img(h5_path, spatial_path, scale_factor_path)
    hi_img = E2I.getImage(k, alpha)
    
    exp_name = os.path.basename(os.path.normpath(visium_dir))
    out_file = f"{out_dir}{exp_name}_k_{k}_alpha_{alpha}.png"
    print(f"Image is saved at {out_file}")
    cv2.imwrite(out_file, hi_img)
