from .Preprocess import Preprocess
from .Embedder import Embedder
from .Embedding2Image import Embedding2Image
from .DimensionReducer import DimensionReducer
from pathlib import Path


class Expression2Img(Preprocess, Embedder, Embedding2Image):
    """Convert visium expresison dat into RGB img"""
    def __init__(self, h5_path: Path, spatial_path: Path, scale_factor_path: Path):
        self._h5_path = h5_path
        self._spatial_path = spatial_path
        self._scale_factor_path = scale_factor_path
        Preprocess.__init__(self)
        Embedder.__init__(self)
        Embedding2Image.__init__(self)
    
    def getImage(self, zdim : int = 256, PEalpha : float = 0.5):
        """Get img converted from visium expression data"""
        print("---- Load data ----")
        # ExpressionData (genes, cell), Spatial (x,y)
        adata, expressionData, spatialMatrix = self.getData(self._h5_path, self._spatial_path, self._scale_factor_path)    
        print("---- Reduce dimensions ----")
        DR = DimensionReducer(expressionData, spatialMatrix, zdim, PEalpha)
        zOut = DR.reduceDims()
        print("---- Get GAE embedding ----")
        z_new = self.getGAEEmbedding(zOut, spatialMatrix)
        adata.obsm["embedding"] = z_new
        print("---- Convert embedding to image ----")
        return self.emb2img(adata)