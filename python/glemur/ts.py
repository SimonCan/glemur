#
# ts.py
#
# Read the time series file and store the data.

import numpy as np

class readTs:
    """
    readTs -- Holds the time series data.
    """
    
    def __init__(self, dataDir = 'data', fileName = 'time_series.dat'):
        """
        Reads and stores the data in 'time_series.dat'.
        
        call signature:
        
          readTs(dataDir = 'data', filename = 'time_series.dat')
          
        Keyword arguments:
        
         *dataDir*:
            Data directory.
            
         *fileName*:
            Name of the time series file.        
        """
        
        try:
            data = np.loadtxt(dataDir + '/' + fileName)
            fd = open(dataDir + '/' + fileName, "r")
            firstLine = fd.readline()
            fd.close()
        except IOError:
            return -1
        
        # create keys for dictionary
        keys = firstLine.split()[1:]
        
        # create attributes and assign data values
        for i in range(len(keys)):
            setattr(self, keys[i], data[:,i])
        
