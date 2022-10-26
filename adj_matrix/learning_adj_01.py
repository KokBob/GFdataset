# -*- coding: utf-8 -*-

import scipy.io
import numpy as np
import pandas as pd
import csv
from csv import reader
import itertools
from numpy import genfromtxt


# def adj_matrix_builder(elements, num_nodes):
'''
:param elements: a .csv file which contains the elements of the FE model.
:param num_nodes: number of nodes.
:return: the adjacency matrix.
'''
num_nodes = 7
A = np.zeros((num_nodes, num_nodes))
elements = './elemts.csv'
with open(elements, 'r') as read_obj:   # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)       # Iterate over each row in the csv using reader object
    for row in csv_reader:              # row variable is a list that represents a row in csv

        node_1 = int(row[0]) - 1
        node_2 = int(row[1]) - 1
        node_3 = int(row[2]) - 1
        # node_4 = int(row[3]) - 1

        A[node_1, node_2] = 1
        A[node_2, node_1] = 1

        A[node_1, node_3] = 1
        A[node_3, node_1] = 1

        # A[node_1, node_4] = 1
        # A[node_4, node_1] = 1

        A[node_2, node_3] = 1
        A[node_3, node_2] = 1

        # A[node_2, node_4] = 1
        # A[node_4, node_2] = 1

        # A[node_3, node_4] = 1
        # A[node_4, node_3] = 1

    # return A
# mam pouze jenom 3 konexe mezi nodama 
# v pripade qudra nodu by tam bylo az 20 
# vtk quadra hexadroni 
# https://vtk.org/doc/nightly/html/classvtkQuadraticHexahedron.html
# casto pouzivane pro nelinearni ulohy 
# %%
num_rows = np.size(A,0)
num_cols = np.size(A,1)

filename = 'adj_beam_01.csv'
with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')

    for i in range(num_rows):

        for j in range(num_cols):
            list = []
            if A[i,j]!=0:
                list.append(j+1)
                list.append(i+1)
                writer.writerow(list)
# %%
'''

:param A: an adjacency matrix
:param filename: the name of the file where the formatted adjacency matrix will be stored.
:return: -
'''

num_rows = np.size(A,0)
num_cols = np.size(A,1)
counter = 0

with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')

    for k in range(num_dir*t_steps):

        for i in range(num_rows):

            for j in range(num_cols):
                list = []
                if A[i,j]!=0:
                    list.append(j+1 + (k * num_nodes))
                    list.append(i+1 + (k * num_nodes))
                    writer.writerow(list)
                    counter = counter + 1
                    print(counter)
