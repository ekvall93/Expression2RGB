from .GAEembedder import GAEembedder
from .AdjacencyMatrix import AdjacencyMatrix
import numpy as np


class Embedder(AdjacencyMatrix, GAEembedder):
    def __init__(self):
        AdjacencyMatrix.__init__(self)
        GAEembedder.__init__(self)

    def _discretize(self, z):
        """Discretize embeddings"""
        return 1.0 * (z > np.mean(z, axis=0))

    def getGAEEmbedding(self, z, spatialMatrix):
        """Get GAE embeddings"""
        adj, edgeList = self.generateAdj(z, spatialMatrix)
        zOutX = self.train(self._discretize(z), adj)
        return zOutX