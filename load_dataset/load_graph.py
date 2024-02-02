# -*- coding: utf-8 -*-
"""
"""
import sys
sys.path.append("..") #
from gnsfem import preprocessing
from pathlib import Path

class graphset(object):
    def __init__(self, input_file_name, endDef):
        d=open(input_file_name,"r").readlines() 
        line_nodes, line_elements, line_end = preprocessing.get_lines_number(d, endDef = endDef )# b2
        self.d_nodes         = preprocessing.get_nodes(input_file_name, line_elements,line_nodes)
        self.d_elements      = preprocessing.get_edges(input_file_name, line_elements,line_nodes, line_end)
        pass
class Plane(object):
    def __init__(self, ):
        self.input_file_name = '../datasets/pl/Plane.inp'  # Plane
        self.path_ = Path(self.input_file_name)
        endDef = '*Nset, nset=RC-1_Set-All-Material, generate'
        self.graph_set = graphset(self.input_file_name, endDef = endDef)
        self.d_nodes = self.graph_set.d_nodes
        self.d_elements = self.graph_set.d_elements
        self.Gcoor = preprocessing.create_graph_coor_tetra(self.d_nodes, self.d_elements) 
        self.G0 = preprocessing.create_graph_coor_tetra(self.d_nodes, self.d_elements)
        lG0 = [self.G0.nodes[tnode].pop('pos',None) for tnode in self.G0.nodes]
class Fibonacci(object):
    def __init__(self, ):
        self.input_file_name = '../datasets/fs/Fibonacci.inp' 
        self.path_ = Path(self.input_file_name)
        endDef = '*Elset, elset=Set-All, generate'
        self.graph_set = graphset(self.input_file_name, endDef = endDef)
        self.d_nodes = self.graph_set.d_nodes
        self.d_elements = self.graph_set.d_elements
        self.Gcoor = preprocessing.create_graph_coor(self.d_nodes, self.d_elements)# C3D8 Hexa
        self.G0 = preprocessing.create_graph_coor(self.d_nodes, self.d_elements)
        lG0 = [self.G0.nodes[tnode].pop('pos',None) for tnode in self.G0.nodes]
class Beam3D(object):
    def __init__(self, ):
        self.input_file_name = '../datasets/b3/Beam3D.inp' 
        self.path_ = Path(self.input_file_name)
        endDef = '*Elset, elset=Set-All, generate'
        self.graph_set = graphset(self.input_file_name, endDef = endDef)
        self.d_nodes = self.graph_set.d_nodes
        self.d_elements = self.graph_set.d_elements
        self.Gcoor = preprocessing.create_graph_coor(self.d_nodes, self.d_elements)# C3D8 Hexa
        self.G0 = preprocessing.create_graph_coor(self.d_nodes, self.d_elements)
        lG0 = [self.G0.nodes[tnode].pop('pos',None) for tnode in self.G0.nodes]
class Beam2D(object):
    def __init__(self, ):
        self.input_file_name = '../datasets/b2/Beam2D.inp' 
        self.path_ = Path(self.input_file_name)
        endDef = '*Nset, nset=Set-Part, generate'
        self.graph_set = graphset(self.input_file_name, endDef = endDef)
        self.d_nodes = self.graph_set.d_nodes
        self.d_elements = self.graph_set.d_elements
        self.Gcoor = preprocessing.create_graph_coor_2D(self.d_nodes, self.d_elements)
        self.G0 = preprocessing.create_graph_coor_2D(self.d_nodes, self.d_elements)
        lG0 = [self.G0.nodes[tnode].pop('pos',None) for tnode in self.G0.nodes]
