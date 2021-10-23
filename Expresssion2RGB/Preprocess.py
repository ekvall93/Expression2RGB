from anndata import AnnData
import scanpy as sc
import pandas as pd
import json
import numpy as np
import torch
from pathlib import Path
from typing import Tuple
from numpy import ndarray
from torch import tensor
from pandas import DataFrame

class Preprocess:
    """Load Visium data and process it"""   
    def getData(self, h5_path: Path, spatial_path: Path, scale_factor_path: Path)->Tuple[AnnData, ndarray, tensor]:
        """Load visium anndata, expresion and spatial data"""
        #Get anndata
        anndata = self._load_data(h5_path, spatial_path, scale_factor_path); self._normalize(anndata)
        #Get expression data
        expression_df = pd.DataFrame(anndata.X.A.T)
        expressionData = self._filterExpressionData(expression_df)
        #Get spatial data
        coords_array = np.array([list(t) for t in zip(anndata.obs["array_row"].tolist(), anndata.obs["array_col"].tolist())])
        spatialMatrix = self._getSpatialData(coords_array)
        return anndata, expressionData, spatialMatrix
    
    def _getSpatialData(self, coords_array: ndarray)->tensor:
        """Get spatial data from coordinate array"""
        spatialMatrix = self._preprocessSpatial(coords_array)
        spatialMatrix = torch.from_numpy(spatialMatrix)
        spatialMatrix = spatialMatrix.type(torch.FloatTensor)
        return spatialMatrix
    
    @staticmethod
    def _preprocessSpatial(originalMatrix: ndarray)->ndarray:
        """
        Preprocess spatial information
        Only works for 2D now, can be convert to 3D if needed
        Normalize all the coordinates to [-0.5,0.5]
        center is [0., 0.]
        D is maximum value in the x/y dim
        """
        spatialMatrix = np.zeros((originalMatrix.shape[0], originalMatrix.shape[1]))
        x = originalMatrix[:, 0]
        y = originalMatrix[:, 1]
        rangex = max(x)-min(x)
        rangey = max(y)-min(y)
        spatialMatrix[:, 0] = (x-min(x))/rangex-0.5
        spatialMatrix[:, 1] = (y-min(y))/rangey-0.5
        return spatialMatrix
    
    @staticmethod
    def _normalize(adata : AnnData)->None:
        """Normalize anndata with log1p"""
        sc.pp.normalize_total(adata,target_sum=1e4)
        sc.pp.log1p(adata)

    @staticmethod
    def _load_data(h5_path: Path, spatial_path: Path, scale_factor_path: Path)->AnnData:
        """Load Visium data and set relevant variables"""
        adata = sc.read_10x_h5(h5_path)
        spatial_all = pd.read_csv(spatial_path, sep=",", header=None, na_filter=False, index_col=0)
        spatial = spatial_all[spatial_all[1] == 1]
        spatial = spatial.sort_values(by=0)
        assert all(adata.obs.index == spatial.index)
        #Rows are genes, and cols are cells
        adata.obs["array_row"], adata.obs["array_col"] = spatial[2], spatial[3]
        adata.obs["pxl_row_in_fullres"], adata.obs["pxl_col_in_fullres"] = spatial[5], spatial[4]
        adata.var_names_make_unique()
        # Read scale_factor_file
        with open(scale_factor_path) as fp_scaler:
            scaler = json.load(fp_scaler)
        adata.uns["tissue_hires_scalef"] = scaler["tissue_hires_scalef"]
        adata.uns["fiducial_diameter_fullres"] = scaler["fiducial_diameter_fullres"]
        return adata

    @staticmethod
    def _filterExpressionData(df: DataFrame, cellRatio: float=1.0, geneRatio: float=0.99, n_genes: int=2000)->ndarray:
        """Filter data by cell and gene ratio, and then pick data by top n_genes"""
        df1 = df[df.astype('bool').mean(axis=1) >= (1-geneRatio)]
        print('After preprocessing, {} genes remaining'.format(df1.shape[0]))
        criteriaGene = df1.astype('bool').mean(axis=0) >= (1-cellRatio)
        df2 = df1[df1.columns[criteriaGene]]
        print('After preprocessing, {} cells have {} nonzero'.format(
            df2.shape[1], geneRatio))
        criteriaSelectGene = df2.var(axis=1).sort_values()[-n_genes:]
        df3 = df2.loc[criteriaSelectGene.index]
        return df3.to_numpy().astype(float)