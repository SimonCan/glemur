#
# data.py
#
# Reads the dump files.

import numpy as np
import pyvtk as vtk
from vtk import vtkStructuredPointsReader, vtkStructuredGridReader
from vtk.util import numpy_support as VN


class readB0:
    """
    readB0 -- Holds the initial magnetic field.
    """
    
    def __init__(self, dataDir = 'data', fileName = 'B0.vtk'):
        """
        Reads the B0 file.
        
        call signature:
        
          readB0(dataDir = 'data', fileName = 'B0.vtk')
          
        Keyword arguments:
        
         *dataDir*:
            Data directory.
            
         *fileName*:
            Name of the file storing B(X,0) = B0.
        """
        
        reader = vtkStructuredPointsReader()

        if (dataDir[-1] == '/'):
            dataDir = dataDir[:-1]
            
        try:
            reader.SetFileName(dataDir + '/' + fileName)
        except IOError:
            return -1
        
        reader.ReadAllVectorsOn()
        reader.ReadAllScalarsOn()
        reader.Update()

        output = reader.GetOutput()
        dim = output.GetDimensions()
        data = output.GetPointData()

        tmp = VN.vtk_to_numpy(data.GetArray('bfield'))
        tmp = tmp.reshape(dim[2], dim[1], dim[0], 3)
        tmp = np.swapaxes(tmp, 0, 3)
        tmp = np.swapaxes(tmp, 1, 2)
        setattr(self, 'bfield', tmp)
        del(tmp)
        
        # meta data from the header
        setattr(self, 'nx', dim[0])
        setattr(self, 'ny', dim[1])
        setattr(self, 'nz', dim[2])
        origin = output.GetOrigin()
        setattr(self, 'x0', origin[0])
        setattr(self, 'y0', origin[1])
        setattr(self, 'z0', origin[2])
        spacing = output.GetSpacing()
        setattr(self, 'dx', spacing[0])
        setattr(self, 'dy', spacing[1])
        setattr(self, 'dz', spacing[2])


class readDump:
    """
    readDump -- Holds the data from a dump file.
    """
    
    def __init__(self, dataDir = 'data', fileName = 'save.vtk'):
        """
        Reads the dump file.
        
        call signature:
        
          readDump(dataDir = 'data', fileName = 'save.vtk')
          
        Keyword arguments:
        
         *dataDir*:
            Data directory.
            
         *fileName*:
            Name of the dump file.
        """
        
        reader = vtkStructuredGridReader()
        
        if (dataDir[-1] == '/'):
            dataDir = dataDir[:-1]
        
        try:
            reader.SetFileName(dataDir + '/' + fileName)
        except IOError:
            return -1
                    
        reader.ReadAllVectorsOn()
        reader.ReadAllScalarsOn()
        reader.Update()

        output = reader.GetOutput()
        dim = output.GetDimensions()
        data = output.GetPointData()
        
        nArrays = data.GetNumberOfArrays()
        for i in range(nArrays):
            arrayName = data.GetArrayName(i)
            tmp = VN.vtk_to_numpy(data.GetArray(arrayName))            
            if (tmp.ndim == 2):
                tmp = tmp.reshape(dim[2], dim[1], dim[0], 3)
                tmp = np.swapaxes(tmp, 0, 3)
                tmp = np.swapaxes(tmp, 1, 2)
            else:
                tmp = tmp.reshape(dim[2], dim[1], dim[0])
                tmp = np.swapaxes(tmp, 0, 2)            
            setattr(self, arrayName, tmp)            
            del(tmp)

        # read the grid coordinates
        points = output.GetPoints()
        grid = VN.vtk_to_numpy(points.GetData())
        grid = grid.reshape(dim[2], dim[1], dim[0], 3)
        grid = np.swapaxes(grid, 0, 3)
        grid = np.swapaxes(grid, 1, 2)
        setattr(self, 'grid', grid)
        del(grid)
        
        # read the parameters
        p = readParams(dataDir = dataDir, fileName = fileName)
        setattr(self, 'p', p)
        
        reader.CloseVTKFile()


class readParams:
    """
    readParams -- Holds the parameters from a dump file.
    """
    
    def __init__(self, dataDir = 'data', fileName = 'save.vtk'):
        """
        Reads the parameters.
        
        call signature:
        
          readParams(dataDir = 'data', fileName = 'save.vtk')
                    
        Keyword arguments:
        
         *dataDir*:
            Data directory.
            
         *fileName*:
            Name of the dump file.
        """
        
        reader = vtkStructuredGridReader()
        
        if (dataDir[-1] == '/'):
            dataDir = dataDir[:-1]
        
        try:
            reader.SetFileName(dataDir + '/' + fileName)
        except IOError:
            return -1
                    
        reader.ReadAllVectorsOn()
        reader.ReadAllScalarsOn()
        reader.Update()

        output = reader.GetOutput()
        data = output.GetPointData()
        
        field = output.GetFieldData()
        nArrays = field.GetNumberOfArrays()
        for i in range(nArrays):
            arrayName = field.GetArrayName(i)
            tmp =  VN.vtk_to_numpy(field.GetArray(arrayName))            
            if (arrayName == 'nx_ny_nz'):
                setattr(self, 'nx', tmp[0]) 
                setattr(self, 'ny', tmp[1]) 
                setattr(self, 'nz', tmp[2])
            elif (arrayName == 'Lx_Ly_Lz'):
                setattr(self, 'Lx', tmp[0]) 
                setattr(self, 'Ly', tmp[1]) 
                setattr(self, 'Lz', tmp[2])
            elif (arrayName == 'Ox_Oy_Oz'):
                setattr(self, 'Ox', tmp[0]) 
                setattr(self, 'Oy', tmp[1]) 
                setattr(self, 'Oz', tmp[2])
            elif (arrayName == 'dx_dy_dz'):
                setattr(self, 'dx', tmp[0]) 
                setattr(self, 'dy', tmp[1]) 
                setattr(self, 'dz', tmp[2])
            elif (arrayName == 'rxhalf_ryhalf'):
                setattr(self, 'rxhalf', tmp[0]) 
                setattr(self, 'ryhalf', tmp[1]) 
            elif (arrayName == 'phi1_phi2'):
                setattr(self, 'phi1', tmp[0]) 
                setattr(self, 'phi2', tmp[1])                 
            else:
                setattr(self, arrayName, tmp[0]) 
                
        reader.CloseVTKFile()
        
           
