# GFdataset
Graph neural network as a new technique of supervised learning is characterised with needs of patterns to learn. 
Following lines describes GF dataset designed for purpose to explore possibilities of hybrid modelling approach connecting  a finite element method and a graph neural network. 

![GFdataset_I](https://user-images.githubusercontent.com/30251196/137585172-d3efe915-0053-43ba-8d05-f592cedd181e.PNG)

## graph path reduction
nx_g = nx.path_graph(5) # a chain 0-1-2-3-4
dgl.from_networkx(nx_g) # from networkx

### refs
https://docs.dgl.ai/guide/graph-external.html
https://networkdata.ics.uci.edu/data.php?id=105
https://networkdata.ics.uci.edu/data.php?id=105
http://konect.cc/networks/ucidata-zachary/
https://github.com/dglai/WWW20-Hands-on-Tutorial/blob/master/basic_tasks/1_load_data.ipynb

