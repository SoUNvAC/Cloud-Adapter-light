L0001: APPENDIX A: Working with the dataset
L0002: 
L0003: 
L0004: The dataset uses numpy arrays to hold the data, therefore the only dependency for reading the data in is to have a python installation
L0005: with the numpy package. We use python 3 and the latest numpy release (1.19.3) but it is unlikely that any versioning issues will arise
L0006: when loading the data.
L0007: 
L0008: To load a subscene array in python using the numpy package:
L0009: 
L0010: 
L0011: 
L0012:    import numpy as np
L0013: 
L0014:    subscene = np.load(<path/to/subscene.npy>)
L0015: 
L0016: 
L0017: 
L0018: To load an associated mask:
L0019: 
L0020: 
L0021: 
L0022:    mask = np.load(<path/to/mask.npy>)
L0023: 
L0024: 
L0025: 
L0026: For an RGB composite of a subscene one would select:
L0027: 
L0028: 
L0029: 
L0030:    RGB = subscene[...,[3,2,1]]
L0031: 
L0032: 
L0033: 
L0034: Several python packages are able to read "classification_tags.csv". We recommend the pandas package as it provides lots of
L0035: functionality for filtering, querying and analysing the data as a DataFrame object (for more information, see here):
L0036: 
L0037: 
L0038: 
L0039:    import pandas as pd
L0040: 
L0041:    table = pd.read_csv("classification_tags.csv",index_col="index")
