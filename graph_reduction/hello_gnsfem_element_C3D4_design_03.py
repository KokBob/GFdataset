from gnsfem import preprocessing
from gnsfem import visualization
import networkx as nx
# original: hello_gnsfem_element_C3D4_design_03.py
#%% select test experiment FILE
file_name   = r'plane_simple.inp'
# file_name   = r'D:/Abaqus/05_Obeam/Obeam.inp'
# file_name   = r'D:/Abaqus/05_Obeam/tetra.inp'
file1       = open(file_name,"r")
d           = file1.readlines() 
#%% get lines number
line_nodes, line_elements, line_end = preprocessing.get_lines_number(d)
d_nodes         = preprocessing.get_nodes(file_name, line_elements,line_nodes)
d_elements      = preprocessing.get_edges(file_name, line_elements,line_nodes, line_end)
G               = preprocessing.create_graph_coor_tetra(d_nodes, d_elements)
# %%
G = G.to_directed(as_view=False)
# %%
G.name = ''
# nx.write_adjlist(G, G.name + ".adjlist")
# nx.set_node_attributes(G, uzly['Label'], "Node Labels FEM")
# %% graphml
# nx.write_graphml(G, G.name + ".graphml")
# nx.draw_networkx(G)
# %% visualization
app= visualization.dashplotly_graph_initial_visualization(d_nodes, G )
if __name__ == "__main__":    
    # app.run_server(debug=True)
    app.run_server(debug=False, port=8052)
# %% graph summary 
num_nodes = len(G.nodes)
num_edges = len(G.edges)
# {}".format(first, second)
print('Summary of graph' + '\n' + 
      '=================' + '\n' 
      'Nodes count:'+ str(num_nodes) + '\n' 
      'Edges count:'+ str(num_edges)
      )
#%% references 
'''
https://www.datacamp.com/community/tutorials/networkx-python-graph-tutorial?utm_source=adwords_ppc&utm_campaignid=898687156&utm_adgroupid=48947256715&utm_device=c&utm_keyword=&utm_matchtype=b&utm_network=s&utm_adpostion=&utm_creative=229765585183&utm_targetid=dsa-429603003980&utm_loc_interest_ms=&utm_loc_physical_ms=1000824&gclid=EAIaIQobChMIq9-J94a26wIVD9myCh3OJA1BEAAYASAAEgJK6PD_BwE
https://www.researchgate.net/post/How_do_I_run_a_python_code_in_the_GPU
https://developer.nvidia.com/blog/gpu-accelerated-graph-analytics-python-numba/
conda install numba cudatoolkit
Finding dense subgraphs via low-rank bilinear optimization
https://dl.acm.org/doi/10.5555/3044805.3045103
gpu netowrkx
https://towardsdatascience.com/4-graph-algorithms-on-steroids-for-data-scientists-with-cugraph-43d784de8d0e
Nvidia Rapids cuGraph: Making graph analysis ubiquitous
https://www.zdnet.com/article/nvidia-rapids-cugraph-making-graph-analysis-ubiquitous/
https://anaconda.org/rapidsai/cugraph
https://www.geeksforgeeks.org/running-python-script-on-gpu/
https://stackoverflow.com/questions/42558165/load-nodes-with-attributes-and-edges-from-dataframe-to-networkx
https://networkx.org/documentation/stable/reference/readwrite/graphml.html
https://networkx.org/documentation/stable/reference/readwrite/generated/networkx.readwrite.graphml.write_graphml.html

'''
