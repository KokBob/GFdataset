# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
from load_dataset import load_dataset 
B2 = load_dataset.Beam2D().get_summary()
B3 = load_dataset.Beam3D().get_summary()
FS = load_dataset.Fibonacci().get_summary()
PL = load_dataset.Plane().get_summary()
