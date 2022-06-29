from tkinter import Label
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy as sc
from shapes import *

'''
This function creates a graph from a list of building blocks by addiing edges between blocks
Input:
    list_shapes : list of shape and arg lists (origin, origin_color, num_nodes)
    start   :   
    graph_type  :   
    graph_args  :   
    add_nodes   :   
    plot    :   
    savefig :   
'''

# Three layers Heterougeneous graph (bottom - up)

def build_lego_structure_from_structure(list_shapes, plot=False, color = 'subgraph', savefig=False, graph_type='nx.connected_watts_strogatz_graph', graph_args=[2,0.4],save2text='',add_node=0):

    G=nx.Graph()
    
    nb_shape=0
    colors=[]   # color of graph for each node (number of graph)
    seen_shapes=[] # shape name dict 
    seen_colors_start=[]  # origin color/index of seen_shape name dict (discontinuous)
    index_roles=[]  # roles for each node (origin color)
    col_start=0 
    label_shape=[]
    
    for shape in list_shapes:
        # updated 1
        if shape[0] in ['Diamond', 'House', 'Hollow']:
            shape_type = shape[0]
        elif shape[0] in ['Chain', 'Clique', 'Cycle', 'Star']:
            shape_type = shape[0] + '_' + str(shape[1] )
        elif shape[0] in ['Tree']:
            shape_type = shape[0] + '_' + str(shape[1]) + '_' + str(shape[2])
        else:
            print('Wrong shape type!')

        # the shape appears for the first time
        if shape_type not in seen_shapes:
            seen_shapes.append(shape_type) 
            if len(index_roles) == 0:
                seen_colors_start.append(0)
            else:
                seen_colors_start.append(np.max([0]+index_roles)+1) 
            col_start=seen_colors_start[-1] 
            ind=len(seen_colors_start)-1
        # the shape has appeared before
        else:
            ind=seen_shapes.index(shape_type)
            col_start=seen_colors_start[ind]
        
        start=len(index_roles)
        # args = origin, origin_color, num_nodes
        args=[start] # origin
        args+=[col_start] # origin_color
        args+=shape[1:] # num_nodes / other params
        graph = eval(shape[0])(*args)

        S = graph.G
        roles = graph.labels
        ### Attach the shape to the basis
        G.add_nodes_from(S.nodes())
        G.add_edges_from(S.edges())
        
        colors+=[nb_shape]*nx.number_of_nodes(S)
        index_roles+=roles
        label_shape+=[col_start]*nx.number_of_nodes(S)
        nb_shape+=1
    #print seen_shapes
    ### Now we link the different shapes:
    N=G.number_of_nodes()
    N_prime=nb_shape
    #### generate Graph
    graph_args.insert(0,N_prime+add_node) # graph_args, add_node are given
    G.add_nodes_from(range(N,N+add_node))
    colors+=[nb_shape+rr for rr in range(add_node)] # for add_node, one color for one node
    #print colors
    r=np.max(index_roles)+1
    l=label_shape[-1]
    index_roles+=[r]*add_node
    label_shape+=[-1]*add_node
    Gg=eval(graph_type)(*graph_args)

    elist=[]
    ### permute the colors:
    initial_col=np.unique(colors)
    perm=np.unique(colors)
    np.random.shuffle(perm)
    color_perm={initial_col[i]:perm[i] for i in range(len(np.unique(colors)))}
    colors2=[color_perm[c] for c in colors]
    #colors=colors2
    for e in Gg.edges():
        if e not in elist:
            ii=np.random.choice(np.where(np.array(colors2)==(e[0]))[0],1)[0]
            jj=np.random.choice(np.where(np.array(colors2)==(e[1]))[0],1)[0]
            G.add_edges_from([(ii,jj)])
            elist+=[e]
            elist+=[(e[1],e[0])]

    if plot==True:
        if color == 'subgraph':
            nx.draw_networkx(G,node_color=colors,cmap="PuRd")
        elif color == 'role':
            nx.draw_networkx(G,node_color=index_roles,cmap="PuRd")
        else: 
            print('Wrong color type')
        
        if savefig==True:
            plt.savefig("plots/structure.png")
    
    if len(save2text)>0:
        graph_list_rep=[['Id','shape_id','type_shape','role']]+\
        [[i,colors[i],label_shape[i],index_roles[i]] for i in range(nx.number_of_nodes(G))]
        np.savetxt(save2text+"graph_nodes.txt",graph_list_rep,fmt='%s,%s,%s,%s')
        elist=[['Source','Target']]+[[e[0],e[1]] for e in G.edges()]
        np.savetxt(save2text+"graph_nodes.txt",graph_list_rep,fmt='%s,%s,%s,%s')
        np.savetxt(save2text+"graph_edges.txt",elist,fmt='%s,%s')
    # return Gg, G, colors, index_roles, label_shape
    return Gg, G, colors, index_roles, label_shape


# Multiple layers homogeneous graph (up - bottom)

def build_fractal_structure(L,graph_type=[],graph_args=[]):
    '''
    builds a hierarchical_structure
    INPUT
    --------------------------------------------------------------------------------------
    L					: nb of layers
    graph_type			: (list of length L) type of graph at every layer (default:graph_type=['nx.gnp_random_graph']*L )
    graph_args			: params for the clusters at each layers
    OUTPUT
    --------------------------------------------------------------------------------------
    '''
    ####Check if all the arguments have been correctly provided or set to default
    if len(graph_type)!=L or len(graph_args)!=L:
        graph_type=['nx.gnp_random_graph']*L
        graph_args=[[10,0.7]]*L
  
    G_h = {}
    labels_h = {}
    G_s = {}
    #G0=nx.gnp_random_graph(level0[0],level0[1])
    G0=eval(graph_type[0])(*graph_args[0])
    labels=[0]*G0.number_of_nodes()

    G_h[0] = G0
    labels_h[0] = labels
    for i in range(1,L):
        G0, Gg, labels=build_new_level(G0,labels,[graph_type[i]],graph_args[i])
        G_h[i] = G0
        labels_h[i] = labels
        G_s[i] = Gg
        #graph_args.remove(0)\
    # print('number of connected components:%i'%(nx.number_connected_components(G0) ))  
    return G_h, G_s, labels_h
        
        
            
def build_new_level(G0,labels=[],graph_type=[],graph_args=[]):
    '''
    builds a hierarchical_structure
    INPUT
    --------------------------------------------------------------------------------------
    L					: nb of layers
    graph_type			: (list of length L) type of graph at every layer (default:ER)
    nb_nodes			: nb of nodes at each level (list of length L)
    graph_args			: params for the clusters at each layers
    OUTPUT
    --------------------------------------------------------------------------------------
    '''
    G=nx.Graph()
    ####Check if all the arguments have been correctly provided or set to default
    if len(graph_type)==0 or len(graph_args)==0:
        graph_type=['nx.gnp_random_graph']
        graph_args=[10,0.7]
    if len(labels)==0:
        labels=[0]*G0.number_of_nodes()
    colors=[]         ## labels for the different shapes
    ### Bottom up construction
    #print(graph_type[0])
    Gg=eval(graph_type[0])(*graph_args)  ####graph structure at new level
    n0=G0.number_of_nodes()
    for i in range(Gg.number_of_nodes()):
        colors+=list(np.array(labels)+i*n0)
        mapping={n: n+i*n0 for n in G0.nodes.keys()}
        G1=nx.relabel_nodes(G0,mapping, copy=True)
        G.add_nodes_from(G1.nodes())
        G.add_edges_from(G1.edges())
    #print(nx.number_connected_components(G0))
    #print('adding links')
    elist=[e for e in G.edges]
    print(elist)
    for e in Gg.edges():
        ii=np.random.choice(np.where( (np.array(colors)//n0)==(e[0]))[0],1)[0]
        jj=np.random.choice(np.where((np.array(colors)//n0)==(e[1]))[0],1)[0]
        if (min([ii,jj]),max([ii,jj])) not in elist:
            #print((min([ii,jj]),max([ii,jj])))
            G.add_edges_from([(ii,jj)])
            elist+=[(min([ii,jj]),max([ii,jj]))]
       
    return G, Gg, colors