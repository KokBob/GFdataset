import pandas as pd
import numpy as np
import networkx as nx
# from numba import jit
import torch
import dgl
import re
# %%
def create_graph_coor_tetra(d_nodes, d_elements):
    G = nx.Graph()
    for nd in d_nodes.index: 
        src0 = d_nodes.iloc[nd]
        src = int(src0['#Node'])
        pos0 = (src0['x'], src0['y'], src0['z'])
        G.add_node(src, pos = pos0)
    for i in range(np.shape(d_elements.index)[0]):
        els = d_elements.iloc[i].values 
        
        ea = els[0:]
        e2 = ea.reshape([2,2]).T
        G.add_edges_from(e2)
        e3 = ea.reshape([2,2])
        G.add_edges_from(e3)
        e4 =np.roll(e3,1)
        G.add_edges_from(e4)
    return G
def create_graph_coor_dgl_2D(d_nodes, d_elements):
    G = nx.Graph()
    # def create_graph_DGL_v2(d_nodes, d_elements):
    G = dgl.DGLGraph() 
    # G.add_nodes(len(d_nodes)-1)
    for nd in d_nodes.index:    
        src0 = d_nodes.iloc[nd]
        src = int(src0['#Node'])
        pos0 = (src0['x'], src0['y'])
        G.add_node(src, pos = pos0)
    for i in range(np.shape(d_elements.index)[0]):
        els0 = d_elements.iloc[i].values 
        els1 = np.roll(els0,1)
        elsf = np.array([els0,els1]).T
        G.add_edges_from(elsf)
    return G
#% create graph connections 
# @jit
def create_graph_coor_2D(d_nodes, d_elements):
    G = nx.Graph()
    for nd in d_nodes.index:    
        src0 = d_nodes.iloc[nd]
        src = int(src0['#Node'])
        pos0 = (src0['x'], src0['y'])
        G.add_node(src, pos = pos0)
    for i in range(np.shape(d_elements.index)[0]):
        els0 = d_elements.iloc[i].values 
        els1 = np.roll(els0,1)
        elsf = np.array([els0,els1]).T
        G.add_edges_from(elsf)
    return G
#% create graph connections 
# @jit

def create_graph_coor(d_nodes, d_elements):
    G = nx.Graph()
    for nd in d_nodes.index:
    
        src0 = d_nodes.iloc[nd]
        src = int(src0['#Node'])
        pos0 = (src0['x'], src0['y'], src0['z'])
        G.add_node(src, pos = pos0)
    
    for i in range(np.shape(d_elements.index)[0]):
        els = d_elements.iloc[i].values 
        ea = els[0:]
        e2 = ea.reshape([2,4]).T
        G.add_edges_from(e2)
        e30 = ea[0:4]
        e31 = np.roll(e30,1)
        er1 = np.array([e30,e31]).T
        G.add_edges_from(er1)
        e40 = ea[4:]
        e41 = np.roll(e40,1)
        er2 = np.array([e40,e41]).T
        G.add_edges_from(er2)
    return G
#% create graph connections 
# @jit
def create_graph_coor_old(d_nodes, d_elements):
#    src=d_nodes['#Node'].values
#    pos=(d_nodes['x'].values,d_nodes['y'].values, d_nodes['z'].values)
    G = nx.Graph()
    for nd in d_nodes.index:

        src0 = d_nodes.iloc[nd]
        src = int(src0['#Node'])
        pos = (src0['x'], src0['y'], src0['z'])
        
        G.add_node(src, pos = pos)

#    for do in d_elements:
    for i in range(np.shape(d_elements.index)[0]):    
#        els = d_elements[do].values # !spatne! potrebuju radek
#        els = d_elements.iloc[do].values  
        els = d_elements.iloc[i].values
        for ei in els:
#            print(str(ei) + ', ' + str(ei + 1))
            G.add_edge(ei,ei + 1)
    G.remove_node(d_nodes.shape[0]+1)
    return G
#% create graph connections 
# @jit
def create_graph(d_nodes, d_elements):
    src=d_nodes['#Node'].values
    G = nx.Graph()
    G.add_nodes_from(src)
#    for do in d_elements:
    for i in range(np.shape(d_elements.index)[0]):
    
#       els = d_elements[do].values # !spatne! potrebuju radek
#        els = d_elements.iloc[do].values 
       els = d_elements.iloc[i].values   
       for ei in els:
#            print(str(ei) + ', ' + str(ei + 1))
            G.add_edge(ei,ei + 1)
    return G

#% create graph connections 
# @jit
def create_graph_old(d_nodes, d_elements):
    src=d_nodes['#Node'].values
    G = nx.Graph()
    G.add_nodes_from(src)
#    for do in d_elements:
    for i in range(np.shape(d_elements.index)[0]):
    
#       els = d_elements[do].values # !spatne! potrebuju radek
#        els = d_elements.iloc[do].values 
       els = d_elements.iloc[i].values   
       for ei in els:
#            print(str(ei) + ', ' + str(ei + 1))
            G.add_edge(ei,ei + 1)
    return G

#% get edges 
def get_edges(file_name, line_elements,line_nodes, line_end):
    lines2read2 = line_end - (line_elements +1)
    d_el_size = pd.read_csv(file_name, skiprows = line_elements +1, nrows=1)
    ar = np.arange(1,d_el_size.size)
    at =list(ar)
    d_elements = pd.read_csv(file_name, names = at, skiprows = line_elements +1, nrows=lines2read2 )
    return d_elements

# get nodes 2D
def get_nodes_2D(file_name, line_elements,line_nodes):
    listNames = ['#Node', 'x', 'y']
    lines2read = line_elements - (line_nodes +1)
    d_nodes = pd.read_csv(file_name, names = listNames, skiprows = line_nodes +1, nrows=lines2read )
    return d_nodes
def get_nodes(file_name, line_elements,line_nodes):
    listNames = ['#Node', 'x', 'y', 'z']
    lines2read = line_elements - (line_nodes +1)
    d_nodes = pd.read_csv(file_name, names = listNames, skiprows = line_nodes +1, nrows=lines2read )
    return d_nodes
def getElementAndNodeLines(linesFromINP, getElementType = False):
    '''
    Example to use
    linesFromINP = Lines
    d = getElementAndNodeLines(linesFromINP)  
    '''
    Lines = linesFromINP
    rx_dict = { 'Stared': re.compile(r'\*$'),
                '1Star': re.compile(r'\*\b'),
                '2Star': re.compile(r'\*\* \b'), }
    dg, dout = {},{}    # dictionary global, dictionary out
    i= 0
    a, b = 0, 0
    seq = []
    for line in Lines:
        try:    dg[matchingLine] = [i, b]
        except: pass
        for key, rx in rx_dict.items():
            match = rx.search(line)
            if match:   
                matchingLine = line.split('\*')[0]
                dg[matchingLine] = [i, b]
                b = i 
        i += 1
    for catch_ in dg:
        if 'Node' in catch_:
            dout[ catch_.split('\n')[0] ] = dg[catch_]
        elif 'Element' in catch_:
            if not getElementType:  dout[catch_.split(', ')[0]] = dg[catch_] 
            else:                   dout[catch_] = dg[catch_] 
    return dout

#% get lines number
def get_lines_number(d, endDef = None):
    if not endDef: endDef = '*End Part'
    for i in range(len(d)):
        if d[i].find('*Node') == 0: 
            line_nodes = i
#            print(i+1)
        elif d[i].find('*Element') == 0: 
            line_elements = i
#            print(i+1)
        # elif d[i].find('*End Part') == 0: # 
        elif d[i].find(endDef) == 0: # 
#            print(i+1)
            line_end = i
        else: pass
    return line_nodes, line_elements, line_end
