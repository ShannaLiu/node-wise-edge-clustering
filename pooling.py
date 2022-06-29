from turtle import forward
from cluster import compute_distance_score
import torch
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_sparse import coalesce
from torch.nn import PairwiseDistance
import torch_geometric.transforms as T
import torch_geometric.utils as U
from torch_geometric.data import Data
import networkx as nx

import sys
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import math
from shapes import *
from build_structure import *
from collections import namedtuple



# score_method : 
# distance: distance of the two embeddings / lin: learned by a linear layer (may or may not reflect dissimilarity)

class dissimpooling(torch.nn.Module):
    unpool_description = namedtuple("UnpoolDescription",
        ["edge_index", "selected_assignment_matrix", "selected_node_score_matrix"])
    def __init__(self, score_method, in_channels, p, normalize, self_add, upper_bound, greedy, select):
        super().__init__()
        self.in_channels = in_channels 
        self.p = p
        self.normalize = normalize 
        self.self_add = self_add
        self.upper_bound = upper_bound
        self.greedy = greedy 
        self.select = select 
        self.score_method = score_method
        if score_method == 'lin':
            self.reset_parameters()
            self.lin = torch.nn.Linear(in_channels, 1)
            def reset_parameters(self):
                self.lin.reset_parameters()
    
    def forward(self, x, edge_index):
        if self.score_method == 'lin':
            edge_score = self.lin(x)
        else: 
            edge_score = self.compute_distance_score(edge_index, x)
        selected_assignment_matrix, selected_node_score_matrix, unpool_info = self.cluster_assignment(edge_score, edge_index)
        adj_new = torch.matmul(torch.matmul(selected_assignment_matrix,U.to_dense_adj(edge_index)[0]),selected_assignment_matrix.t()) 
        adj_new[adj_new>0] = 1
        edge_index_new = U.remove_self_loops(U.dense_to_sparse(adj_new)[0])[0]
        x_new = torch.matmul(selected_node_score_matrix, x)
        return edge_index_new, x_new, selected_assignment_matrix, selected_node_score_matrix, unpool_info

    # return selected_asignment_matrix and selected_node_score_matrix for future relational integrator

    # Compute the score for each edge based on the distance
    def compute_distance_score(self, edge_index=[], x=[]):
            pdist = PairwiseDistance(self.p)
            dist = pdist(x[edge_index[0]], x[edge_index[1]])
            
            if self.normalize == True:
                dist_norm = (dist - torch.min(dist)) / (torch.max(dist) - torch.min(dist))
                dist_norm = dist_norm + self.self_add
            else:
                dist_norm = dist
            return dist_norm

    def node__cluster_search(self, edge_score, edge_index, i):
        selected_score = []
        in_cluster = [i]
        selected_edge = torch.tensor([])
        while (sum(selected_score) < self.upper_bound):
            potential_edge_index = torch.tensor([])
            potential_score = torch.tensor([])
            for i in in_cluster:
                # the first element in edge is in cluster
                potential_edge_index = torch.cat([potential_edge_index, edge_index[:,edge_index[0]==i]], dim=-1) 
                potential_score = torch.cat([potential_score, edge_score[edge_index[0]==i]], dim=-1)
                for j in in_cluster:
                    potential_score = potential_score[potential_edge_index[1]!=j]
                    potential_edge_index = potential_edge_index[:,potential_edge_index[1]!=j]
            if (potential_score.shape[0]==0):
                print('The current cluster has contained all nodes in the graph')
                break
            added_score = torch.min(potential_score)
            added_edge = potential_edge_index[:,torch.where(potential_score==added_score)]
            selected_edge = torch.cat([selected_edge, added_edge], dim=-1)   
            selected_score += [added_score.item()]
            in_cluster += [int(added_edge[1].item())]
        if (self.greedy==False):
            in_cluster = in_cluster[0:(len(in_cluster)-1)]
            selected_edge = selected_edge[:,0:(selected_edge.shape[1]-1)]
            selected_score = selected_score[0:(len(selected_score)-1)]

        return in_cluster, selected_edge, selected_score

    def cluster_assignment(self, edge_score, edge_index):
        num_nodes = torch.max(edge_index) + 1
        assignment_matrix = torch.tensor([])
        node_score_matrix = torch.tensor([])
        cluster_score = torch.tensor([])
        # cluster assignment matrix (C*N C: number of considered clusters, N: number of nodes)
        # 

        num_cluster = num_nodes 
        assignment_matrix = torch.zeros((num_cluster, num_nodes))
        node_score_matrix = torch.zeros([num_cluster, num_nodes])
        cluster_score = torch.zeros([num_cluster])

        for i in range(num_cluster):
            in_cluster, selected_edge, selected_score = self.node__cluster_search(edge_score, edge_index, i)
            assignment_matrix[i,in_cluster] = 1
            cluster_score[i] = sum(selected_score)
            for j in range(len(selected_score)):
                node_score_matrix[i,int(selected_edge[0,j].item())] += selected_score[j]/2
                node_score_matrix[i,int(selected_edge[1,j].item())] += selected_score[j]/2

        # delete repeated cluster
        # bug: cluster may have same assignment, but different score
        _, index, _ = np.unique(assignment_matrix.numpy(), return_index=True, return_inverse=True, axis=0)
        assignment_matrix = assignment_matrix[index,]
        node_score_matrix = node_score_matrix[index,]
        cluster_score = cluster_score[index]
        
        if self.select == True:
            selected_assignment_matrix = torch.tensor([])
            selected_node_score_matrix = torch.tensor([])
            potential_assignment_matrix = assignment_matrix
            potential_node_score_matrix = node_score_matrix
            potential_cluster_score = cluster_score
            potential_node_index = torch.ones(num_nodes) # node index that hasn't been selected
            while (torch.sum(torch.sum(selected_assignment_matrix, dim=0)>0) < num_nodes):
                num_of_element = torch.sum(potential_assignment_matrix[:,potential_node_index>0], dim=1)
                max_cluster_score = torch.max(potential_cluster_score[torch.where(num_of_element==torch.max(num_of_element))])
                selected_cluster_index = torch.where(potential_cluster_score==max_cluster_score)[0]
                added_assignment_matrix = potential_assignment_matrix[selected_cluster_index,:]
                added_node_score_matrix = potential_node_score_matrix[selected_cluster_index,:]
                selected_assignment_matrix = torch.cat([selected_assignment_matrix, added_assignment_matrix], dim=0)
                potential_assignment_matrix = torch.cat([potential_assignment_matrix[0:selected_cluster_index,:],potential_assignment_matrix[selected_cluster_index+1:,:]],dim=0)
                selected_node_score_matrix = torch.cat([selected_node_score_matrix, added_node_score_matrix], dim=0)
                potential_node_score_matrix = torch.cat([potential_node_score_matrix[0:selected_cluster_index,:],potential_node_score_matrix[selected_cluster_index+1:,:]],dim=0)
                potential_node_index[torch.sum(selected_assignment_matrix, dim=0)>0] = 0
        elif self.select==False:
            selected_assignment_matrix = assignment_matrix
            selected_node_score_matrix = node_score_matrix
            
        unpool_info = self.unpool_description(edge_index=edge_index,
                                              selected_assignment_matrix=selected_assignment_matrix,
                                              selected_node_score_matrix=selected_node_score_matrix)
        return selected_assignment_matrix, selected_node_score_matrix, unpool_info
    
    # Here x is the pooled embedding, x_new is the unpooled embedding
    def unpool(self, x, unpool_info):
        x_new = torch.matmul(torch.nan_to_num(1/unpool_info.selected_node_score_matrix.t(),nan=0.0, posinf=0.0, neginf=0.0), x)
        return x_new, unpool_info.edge_index
