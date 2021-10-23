from pathlib import Path
import os
from dataclasses import dataclass

@dataclass
class VisiumPaths:
    """Class for keeping track of an item in inventory."""
    h5_path: Path = "filtered_feature_bc_matrix.h5"
    spatial_path: Path = "spatial/tissue_positions_list.csv"
    scale_factor_path: Path = "spatial/scalefactors_json.json"

class PathHandler:
    @staticmethod
    def validatePath(path: Path)->Path:
        """Validate path"""
        if not path.endswith("/"):
            path = path + "/"
        return path

    @staticmethod
    def checkFile(file: Path):
        return os.path.isfile(file) 

    @staticmethod
    def checkDir(dir_path: Path):
        return os.path.isdir(dir_path) 