#
# gm2pc.py
#
# Conversion routines from glemur to pencil code.

import numpy as np
import glemur as gm
from scipy.interpolate import griddata
import vtk as vtk
from vtk.util import numpy_support as VN
import struct

def gm2pc(dataDir = 'data', gmFile = 'save.vtk', pcFile = 'gm2pc.vtk', method = 'linear', nxyz = []): 
    """
    Converts glemur data into PencilCode data.
    
    call signature:
    
        gm2pc(dataDir = 'data', gmFile = 'save.vtk', pcFile = 'gm2pc.vtk', method = 'nn', nxyz = []):
    
    Keyword arguments:
    
    *dataDir*:
      Data directory.
        
    *gmFile*:
        Name of the glemur data file.
           
    *pcFile*:
        Name of the PencilCode data file.

    *nxyz*:
      List with number of grid points in x, y and z direction.
      Overwrites default grid.
    """
    
    # read the glemur data
    if (gmFile == 'B0.vtk'):
        data = gm.readB0(dataDir = dataDir)
        bb = data.bfield.transpose([1,2,3,0]).ravel()
        p = gm.readParams(dataDir = dataDir)
        x = np.arange(p.Ox-p.dx, p.Ox+p.Lx+2*p.dx, p.dx)
        y = np.arange(p.Oy-p.dy, p.Oy+p.Ly+2*p.dy, p.dy)
        z = np.arange(p.Oz-p.dz, p.Oz+p.Lz+2*p.dz, p.dz)
        xx = np.array(np.meshgrid(x, y, z))
        xx = np.swapaxes(xx, 1, 2)
        xx = xx.transpose([1,2,3,0]).ravel()
        xx = xx.reshape((xx.shape[0]/3,3))
        bb = bb.reshape((bb.shape[0]/3,3))
    else:
        data = gm.readDump(dataDir = dataDir, fileName = gmFile)
        p = data.p
        xx = data.grid.transpose([1,2,3,0]).ravel()
        bb = data.bfield.transpose([1,2,3,0]).ravel()
        xx = xx.reshape((xx.shape[0]/3,3))
        bb = bb.reshape((bb.shape[0]/3,3))
    
    REAL_STR = 'float'
    if (data.bfield.dtype == 'float64'): REAL_STR = 'double'
    
    # prepare the target arrays
    if (np.array(nxyz).shape[0] == 3):
        nx = nxyz[0]; ny = nxyz[1]; nz = nxyz[2]
        dx = p.Lx/(nx-1); dy = p.Ly/(ny-1); dz = p.Lz/(nz-1)
    else:
        nx = p.nx-2; ny = p.ny-2; nz = p.nz-2
        dx = p.dx; dy = p.dy; dz = p.dz
    x = np.arange(p.Ox, p.Ox+p.Lx+dx/2., dx)
    y = np.arange(p.Oy, p.Oy+p.Ly+dy/2., dy)
    z = np.arange(p.Oz, p.Oz+p.Lz+dz/2., dz)
    xNew = np.array(np.meshgrid(x, y, z))
    xNew = np.swapaxes(xNew, 1, 2)
    
    # interpolate the magnetic field on the regular grid
    bNew = griddata(xx, bb, (xNew[0,...].flatten(), xNew[1,...].flatten(), xNew[2,...].flatten()), method = method)
    bNew = np.reshape(bNew, (nx,ny,nz,3))
    bNew = np.swapaxes(bNew, 0, 2)
    
    # write the data into a vtk file
    f = open(dataDir + '/' + pcFile, 'wb')
    f.write("# vtk DataFile Version 2.0\n")
    f.write("GLEMuR to PC interpolated data\n")
    f.write("BINARY\n")
    f.write("DATASET STRUCTURED_POINTS\n")
    f.write("DIMENSIONS {0:9} {1:9} {2:9}\n".format(nx, ny, nz))
    f.write("ORIGIN {0:12.8} {1:12.8} {2:12.8}\n".format(p.Ox, p.Oy, p.Oz))
    f.write("SPACING {0:12.8} {1:12.8} {2:12.8}\n".format(dx, dy, dz))
    f.write("POINT_DATA {0:9}\n".format(nx*ny*nz))
    f.write("VECTORS bfield {0}\n".format(REAL_STR))
    
    if (REAL_STR == 'float'): f.write(struct.pack('>%sf' %bNew.size, *bNew.flatten()))
    else: f.write(struct.pack('>%sd' %bNew.size, *bNew.flatten()))
    
    f.close()    
    
    return bNew
    
