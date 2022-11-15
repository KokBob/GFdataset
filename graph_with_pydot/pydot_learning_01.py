# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 10:10:41 2022

@author: CIM2BJ
"""

import pydot
import os
from IPython.display import Image, display

#Return current working directory including the path
# rootDir = os.getcwd()
rootDir = r'C:\CAE\dummies\dummies'

#Create a graph object of directed type
G = pydot.Dot(graph_type = "digraph")

#Get the name of the current directory only
currentDir = rootDir.split("/")[-1]

#Add a node named after current directory filled with green color
node = pydot.Node(currentDir, style = "filled", fillcolor = "green")
G.add_node(node)
# %%
#Loop through the directory, subdirectory and files of root directory
for root, dirs, files in os.walk(rootDir):
    
    #Ignore hidden files and folder
    # if root==rootDir or (root.split("/")[6].startswith(".") == False):
       
    for subdir in dirs:
        
        # Ignore hidden folder
        if subdir.startswith(".") == False:
            
            # Add nodes with name of subdirectory and fill it with yellow color
            node = pydot.Node(subdir, style = "filled", fillcolor = "yellow")
            G.add_node(node)

            # Add the edge between root directory and sub directory
            edge = pydot.Edge(root.split("/")[-1], subdir)
            G.add_edge(edge)
        
        for file in files:
            
            #Add node for each file and fill it with orange color
            node = pydot.Node(file, style = "filled", fillcolor = "orange")
            G.add_node(node)

            #Add edge between directory/subdirectory and file 
            edge = pydot.Edge(root.split("/")[-1], file)
            G.add_edge(edge)

#Create the image of folder tree
# %%
im = Image(G.create_jpeg())

#Display the image
# display(im)
# p

#Save the image in jpeg format
G.write_jpeg("folder_tree.jpeg")
