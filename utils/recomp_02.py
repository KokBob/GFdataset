# -*- coding: utf-8 -*-
'''
[2] https://stackoverflow.com/questions/42171709/creating-pandas-dataframe-from-a-list-of-strings
[1] https://stackoverflow.com/questions/53022325/python-all-the-lines-and-line-numbers-in-which-string-occurs-in-the-input-file
[0] https://stackoverflow.com/questions/67387814/multiline-regex-match-retrieving-line-numbers-and-matches
'''
import re
import io
import pandas as pd
# nameRes = 'dsallCF2'
nameRes = 'dsallRF2'
resFile = nameRes + '.rpt'
# resFile = 'dsall.rpt'
# resFile = 'dsall_reduced.rpt'
class rege(object):
    def __init__(self, resFile):
        file1 = open(resFile, "r")
        self.Lines = file1.readlines()
    def rgx_dct_01(self,):
        
        self.rx_dict = { 
            # 'Section' : re.compile (r'-------------------------------------------------------------------------------------------'),
            'Source': re.compile(r'Source'), 
            # 'Element': re.compile(r'  Element Label            Int'), 
            'Process': re.compile(r'  Minimum '), }
    def lines_cathing(self,):
        Lines = self.Lines
        rx_dict = self.rx_dict
        rd = {}
        i= 0
        a, b = 0, 0
        seq = []

        for line in Lines:
            try:    rd[line] = [i, b]
            except: pass
            for key, rx in rx_dict.items():
                match = rx.search(line)
                if match:   
                    # print(match)
                    matchingLine = line.split(r'\s*')[0]
                    # https://stackoverflow.com/questions/4309684/split-a-string-with-unknown-number-of-spaces-as-separator-in-python
                    # blank space separator
                    # matchingLine = line.split()[0]
                    rd[matchingLine] = [i, b]
                    b = i 
                    # print(b)
            i += 1        
        self.rd = rd
        self.b = b
# %%
rg = rege(resFile)
rg.rgx_dct_01()
# %%
# https://stackoverflow.com/questions/40054733/python-extract-multiple-float-numbers-from-string
Lines = rg.Lines
rx_dict = rg.rx_dict
rd = { }
i = 0
block_counter = 0
for line in Lines:
    for key, rx in rx_dict.items():
        match = rx.search(line) 
        if match:   
            if key == 'Source': block_start = i          
            elif key == 'Process':
                block_end = i               
                rd[block_counter] = [block_start, block_end]
                block_counter += 1
    i += 1       
rd_ = rd[125]
L = Lines[rd_[0]+18: rd_[0]+3+27]
def getFloats(line_blockOfFloats):
    s = line_blockOfFloats
    p = re.compile(r' \d+\.')  # Compile a pattern to capture float values
    floats = [float(i) for i in p.findall(s)]  # Convert strings to float
    return floats
# F = list(map(getFloats, L))

# %%
df = pd.DataFrame()

num_elements = 7
num_nodes = 5
cf_line = 16
# cf_line = 19 #smax
# rd_ = rd[125]
for r_ in rd:
    rd_ = rd[r_]
    # L = Lines[rd_[0]+cf_line : rd_[0]+cf_line +num_elements]         
    L = Lines[rd_[0]+cf_line : rd_[0]+cf_line +num_nodes]  
    # L = Lines[rd_[0]+19: rd_[0]+19+num_elements]
    df_test = pd.read_csv(io.StringIO('\n'.join(L)), delim_whitespace=True, header = None)
    df_test = df_test.drop(columns=[0, ]) # Nodes rf etc
    # df_test = df_test.drop(columns=[0, 1])
    ## step_name = Lines[rd_[0]+4].split(':')[1].split('\n')[0] 
    step_name = 'I' + Lines[rd_[0]+5].split(':')[1].split('\n')[0].split(' ')[-1]
    df[step_name] = df_test.max(axis=1)
    print(step_name)
    # break
# %%
df.to_csv(nameRes + '_01.csv')

