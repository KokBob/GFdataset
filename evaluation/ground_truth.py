import pandas as pd
class GroundTruth(object):
    def __init__(self, ):   pass

    def fibonacci(self, vtu_file   = 'fibonacci3D1.vtu'): 
        
        self.vtu_file   = vtu_file    
        
        self.nodes_pd   = pd.read_csv('../evaluation/nodes_Beam3D.csv')
        self.nodes      = self.nodes_pd[['X', 'Y', 'Z']]
        self.xyz        = list(self.nodes.values)

        self.elements           = pd.read_csv('../evaluation/elements_Beam3D.csv')
        self.elements           = self.elements.loc[:, self.elements.columns != 'ID']
        self.elements           = self.elements -min(self.elements.min().values)
        self.elements['count']  =  self.elements.shape[1]
        df_offset               = self.elements['count']
        del self.elements['count']
        self.df_offset          = df_offset.cumsum()     
        
        self.elements['type']   =12 # linear hexa
        self.df_type            = self.elements['type']
        del self.elements['type']
        
        self.points_01 = [item for sublist in self.xyz for item in sublist]
              
    def beam3D(self, vtu_file   = 'beam3D1.vtu'): 
        
        self.vtu_file   = vtu_file    
        
        self.nodes_pd   = pd.read_csv('../evaluation/nodes_Beam3D.csv')
        self.nodes      = self.nodes_pd[['X', 'Y', 'Z']]
        self.xyz        = list(self.nodes.values)

        self.elements           = pd.read_csv('../evaluation/elements_Beam3D.csv')
        self.elements           = self.elements.loc[:, self.elements.columns != 'ID']
        self.elements           = self.elements -min(self.elements.min().values)
        self.elements['count']  =  self.elements.shape[1]
        df_offset               = self.elements['count']
        del self.elements['count']
        self.df_offset          = df_offset.cumsum()     
        
        self.elements['type']   =12 # linear hexa
        self.df_type            = self.elements['type']
        del self.elements['type']
        
        self.points_01 = [item for sublist in self.xyz for item in sublist]
        
    def attach_result_fields(self, X, y, y_hat, err):
        self.point_values_X     =  X
        self.point_values_y     =  y
        self.point_values_y_hat =  y_hat
        self.point_values_err   =  err
    def write_results_to_vtu(self,):
        l1 = '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian"> \n'
        l2 = '<UnstructuredGrid> \n'
        l3 = '<Piece NumberOfPoints="' + str(len(self.nodes)) + '" NumberOfCells="'+ str(len(self.elements)) +'"> \n'
        l4 = '<Points> \n'
        l5 = '<DataArray type="Float64" NumberOfComponents="3" format="ascii"> \n'
        l6 =  self.nodes.to_string(header=False, index=False) 
        l7 = '\n</DataArray> \n'
        l8 = '</Points> \n'
        l81 = '<PointData Tensors="" Vectors="" Scalars="">'
        l82 = '\n<DataArray type="Float32" Name="X_" format="ascii"> \n'
        # l83 = pd.DataFrame(point_values).to_string(header=False, index=False)
        l83 = pd.DataFrame(self.point_values_X).to_string(header=False, index=False)
        l84 = '\n</DataArray> \n'
        
        l82b = '\n<DataArray type="Float32" Name="Y_" format="ascii"> \n'
        l83b = pd.DataFrame(self.point_values_y).to_string(header=False, index=False)
        l84b = '\n</DataArray> \n'
        
        l82c = '\n<DataArray type="Float32" Name="Yhat_" format="ascii"> \n'
        l83c = pd.DataFrame(self.point_values_y_hat).to_string(header=False, index=False)
        l84c = '\n</DataArray> \n'
        
        l82d = '\n <DataArray type="Float32" Name="err_" format="ascii"> \n'
        l83d = pd.DataFrame(self.point_values_err).to_string(header=False, index=False)
        l84d = '\n</DataArray> \n'
        
        l85 = '</PointData> \n' 
        l9 = '<Cells> \n'
        l10 = '<DataArray type="Int32" Name="connectivity" format="ascii"> \n'
        l11 =  self.elements.to_string(header=False, index=False) 
        l11a = '\n</DataArray>\n'
        l12 = '\n<DataArray type="Int32" Name="offsets" format="ascii">\n'
        l13 = self.df_offset.to_string(header=False, index=False)
        l14 = '\n</DataArray>\n'
        l15 = '<DataArray type="UInt8" Name="types" format="ascii">\n'
        l16 = self.df_type.to_string(header=False, index=False) 
        l17 ='\n</DataArray>\n'
        l18 ='</Cells>\n'
        l19 ='</Piece>\n'
        l20 ='</UnstructuredGrid>\n'
        l21 ='</VTKFile>\n'
        
        self.lines = [l1, l2,l3, l4, l5, l6, l7, l8,
                 l81,
                 l82,l83,l84,         
                 l82b,l83b,l84b,
                 l82c,l83c,l84c,
                 l82d,l83d,l84d,
                 l85,           
                 l9, l10, l11,l11a,
                 l12, l13, l14, l15, l16,l17,l18,l19,l20,l21]
                # pass
        with open(self.vtu_file, 'w') as f:     f.writelines(self.lines)

            
