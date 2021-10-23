import numpy as np
from scipy.spatial import distance
import networkx as nx

class AdjacencyMatrix:
    def generateAdj(self, featureMatrix, spatialMatrix):
        """Generating edgeList"""
        edgeList = self.calculateSpatialMatrix(featureMatrix, spatialMatrix)
        graphdict = self.edgeList2edgeDict(edgeList, featureMatrix.shape[0])
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graphdict))
        return adj, edgeList

    def calculateSpatialMatrix(self, featureMatrix, spatialMatrix, distanceType='euclidean', k=6):
        """Calculate spatial Matrix directly use X/Y coordinates"""       
        edgeList=[]
        ## Version 2: for each of the cell, calculate dist, save memory 
        for i in np.arange(spatialMatrix.shape[0]):
            tmp=spatialMatrix[i,:].reshape(1,-1)
            distMat = distance.cdist(tmp,spatialMatrix, distanceType)
            res = distMat.argsort()[:k+1]
            tmpdist = distMat[0,res[0][1:k+1]]
            boundary = np.mean(tmpdist)+np.std(tmpdist) #optional
            for j in np.arange(1,k+1):
                edgeList.append((i,res[0][j],1.0))
                
        return edgeList

    def edgeList2edgeDict(self, edgeList, nodesize):
        graphdict={}
        tdict={}

        for edge in edgeList:
            end1 = edge[0]
            end2 = edge[1]
            tdict[end1]=""
            tdict[end2]=""
            if end1 in graphdict:
                tmplist = graphdict[end1]
            else:
                tmplist = []
            tmplist.append(end2)
            graphdict[end1]= tmplist

        #check and get full matrix
        for i in range(nodesize):
            if i not in tdict:
                graphdict[i]=[]

        return graphdict