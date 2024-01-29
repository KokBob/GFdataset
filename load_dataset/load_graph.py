# -*- coding: utf-8 -*-
"""
"""
import sys
sys.path.append("..") #
from gnsfem import preprocessing

class graphset(object):
    def __init__(self, input_file_name, endDef):
        d=open(input_file_name,"r").readlines() 
        line_nodes, line_elements, line_end = preprocessing.get_lines_number(d, endDef = endDef )# b2
        self.d_nodes         = preprocessing.get_nodes(input_file_name, line_elements,line_nodes)
        self.d_elements      = preprocessing.get_edges(input_file_name, line_elements,line_nodes, line_end)
        pass
    

class Beam2D(object):
    def __init__(self, ):
        self.input_file_name = '../datasets/b2/Beam2D.inp' 
        endDef = '*Nset, nset=Set-Part, generate'
        self.graph_set = graphset(self.input_file_name, endDef = endDef)
        self.d_nodes = self.graph_set.d_nodes
        self.d_elements = self.graph_set.d_elements
        self.Gcoor = preprocessing.create_graph_coor_2D(self.d_nodes, self.d_elements)
        self.G0 = preprocessing.create_graph_coor_2D(self.d_nodes, self.d_elements)
        lG0 = [self.G0.nodes[tnode].pop('pos',None) for tnode in self.G0.nodes]
# Test
# test = Beam2D()
# G = test.Gcoor
# G0 = test.G0
