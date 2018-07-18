#
# poincare.py
#
# Creates Poincare maps.

import numpy as np
import glemur as gm
import pyvtk as vtk
from vtk.util import numpy_support as VN
from os import listdir
import natsort

class poincareMap:
    """
    poincareMap -- Holds the Points of the Poincare map.
    """
    
    def __init__(self, dataDir = 'data', B0File = 'B0.vtk', poincare = 'poincareInit.vtk', paramFile = 'save.vtk', dumpFile = [], interpolation = 'weighted', integration = 'simple', hMin = 2e-3, hMax = 2e2, lMax = 500, tol = 2e-3, iterMax = 1e3, x0 = [], y0 = [], nSeeds = 10, nIter = 10):
        """
        Creates, saves and returns the initial streamline map for a few lines and the Poincare map.
        
        call signature:
        
          streamInit(dataDir = 'data', B0File = 'B0.vtk', poincare = 'poincareInit.vtk', paramFile = 'save.vtk', dumpFile = [], interpolation = 'weighted', integration = 'simple', sub = 1, hMin = 2e-3, hMax = 2e2, lMax = 500, tol = 2e-3, iterMax = 1e3, x0 = [], y0 = [])
        
        Trace magnetic streamlines for t = 0 and store the positions in a file.
        
        Keyword arguments:
        
         *dataDir*:
            Data directory.
            
         *B0File*:
            Name of the file storing B(X,0) = B0.
            
         *poincare*:
            Store the Poincare map in this file.
             
         *paramFile*:
            Dump file from where to read the simulation parameters.
         
         *dumpFile*:
            If not empty map the streamlines from B0.vtk to the time of dumpFile.
            
         *interpolation*:
            Interpolation of the vector field.
            'mean': takes the mean of the adjacent grid point.
            'weighted': weights the adjacent grid points according to their distance.
       
         *integration*:
            Integration method.
            'simple': low order method.
            'RK6': Runge-Kutta 6th order.
       
         *hMin*:
            Minimum step length for and underflow to occur.
        
         *hMax*:
            Parameter for the initial step length.
        
         *lMax*:
            Maximum length of the streamline. Integration will stop if l >= lMax.
        
         *tol*:
            Tolerance for each integration step. Reduces the step length if error >= tol.
         
         *iterMax*:
            Maximum number of iterations.     
         
         *x0*:
            Initial seed in x-direction.
            
         *y0*:
            Initial seed in y-direction.
            
         *nSeeds*:
            Number of initial seeds.
            
         *nIter*:
            Number of times the streamlines go through the domain.
        """
        
        p = gm.readParams(dataDir = dataDir, fileName = paramFile)
        
        if (len(x0) != len(y0)):
            print("error: length of x0 != length of y0.")
            return -1
        if ((len(x0) == 0) and (len(y0) == 0)):
            x0 = np.random.random(nSeeds)*p.Lx + p.Ox
            y0 = np.random.random(nSeeds)*p.Ly + p.Oy
        else:
            nSeeds = len(x0)
        
        x = np.zeros((nIter+1,nSeeds))
        y = np.zeros((nIter+1,nSeeds))
        x[0,:] = x0
        y[0,:] = y0
        streams = range(nSeeds)
        tracers = range(nSeeds)
        sl      = np.zeros(nSeeds, dtype = 'int32')
        for q in range(len(x0)):
            tracers[q] = []
            for i in range(nIter):
                # stream the lines until they hit the upper boundary
                s = gm.streamInit(dataDir = dataDir, B0File = B0File, streamFileInit = [], paramFile = paramFile, interpolation = interpolation, integration = integration, hMin = hMin, hMax = hMax, lMax = lMax, tol = tol, iterMax = iterMax, x0 = x[i,q], y0 = y[i,q])
                
                # in case the initial time is not 0, map the streamline accordingly
                if (dumpFile != []):
                    s = gm.mapStream(dataDir = dataDir, streamFileInit = 'streamInit.vtk', streamFile = [], dumpFile = dumpFile, interpolation = interpolation, s = s)
                
                # interpolate the final values for points hitting through the boundary
                l = (p.Oz + p.Lz - s.tracers[2,0,0,s.sl-2]) / (s.tracers[2,0,0,s.sl-1] - s.tracers[2,0,0,s.sl-2])
                x[i+1,q] = (s.tracers[0,0,0,s.sl-1] - s.tracers[0,0,0,s.sl-2])*l + s.tracers[0,0,0,s.sl-2]
                y[i+1,q] = (s.tracers[1,0,0,s.sl-1] - s.tracers[1,0,0,s.sl-2])*l + s.tracers[1,0,0,s.sl-2]
                
                # add the tracers to the existing ones
                tracers[q].extend(list(np.swapaxes(s.tracers[:,0,0,:s.sl], 0, 1)))
                sl[q] += s.sl
        
        p = s.p
        self.p = p
        self.x = x
        self.y = y
        self.sl = sl
        maxLen = sl.max()        
        self.tracers = np.zeros((3, nSeeds, maxLen))
        for q in range(len(x0)):
            self.tracers[:,q,:sl[q]] = np.swapaxes(np.array(tracers[q]), 0, 1)
        
        # streamline parameters
        self.hMin = hMin
        self.hMax = hMax
        self.lMax = lMax
        self.tol = tol
        self.iterMax = iterMax
        self.nSeeds = nSeeds
        self.nIter = nIter
        
        # safe in vtk file
        if (poincare != []):
            writer = vtk.vtkPolyDataWriter()
            writer.SetFileName(dataDir + '/' + poincare)
            polyData = vtk.vtkPolyData()
            fieldData = vtk.vtkFieldData()
            # fields containing x and y values
            field = VN.numpy_to_vtk(self.x)
            field.SetName('x')
            fieldData.AddArray(field)
            field = VN.numpy_to_vtk(self.y)
            field.SetName('y')
            fieldData.AddArray(field)
            # field containing length of stream lines for later decomposition
            field = VN.numpy_to_vtk(self.sl)
            field.SetName('sl')
            fieldData.AddArray(field)
            # streamline parameters
            tmp = range(10)            
            tmp[0] = np.array([s.hMin], dtype = 'float32'); field = VN.numpy_to_vtk(tmp[0]); field.SetName('hMin'); fieldData.AddArray(field)
            tmp[1] = np.array([s.hMax], dtype = 'float32'); field = VN.numpy_to_vtk(tmp[1]); field.SetName('hMax'); fieldData.AddArray(field)
            tmp[2] = np.array([s.lMax], dtype = 'float32'); field = VN.numpy_to_vtk(tmp[2]); field.SetName('lMax'); fieldData.AddArray(field)
            tmp[3] = np.array([s.tol], dtype = 'float32'); field = VN.numpy_to_vtk(tmp[3]); field.SetName('tol'); fieldData.AddArray(field)
            tmp[4] = np.array([s.iterMax], dtype = 'int32'); field = VN.numpy_to_vtk(tmp[4]); field.SetName('iterMax'); fieldData.AddArray(field)
            tmp[5] = np.array([nSeeds], dtype = 'int32'); field = VN.numpy_to_vtk(tmp[5]); field.SetName('nSeeds'); fieldData.AddArray(field)
            tmp[6] = np.array([nIter], dtype = 'int32'); field = VN.numpy_to_vtk(tmp[6]); field.SetName('nIter'); fieldData.AddArray(field)
            # fields containing simulation parameters stored in paramFile
            dic = dir(p)
            params = range(len(dic))
            i = 0
            for attr in dic:
                if( attr[0] != '_'):
                    params[i] = getattr(p, attr)
                    params[i] = np.array([params[i]], dtype = type(params[i]))
                    field = VN.numpy_to_vtk(params[i])
                    field.SetName(attr)
                    fieldData.AddArray(field)
                    i += 1
            # all streamlines as continuous array of points
            points = vtk.vtkPoints()
            for q in range(len(x0)):
                for l in range(self.sl[q]):
                    points.InsertNextPoint(self.tracers[:,q,l])
            polyData.SetPoints(points)
            polyData.SetFieldData(fieldData)
            writer.SetInput(polyData)
            writer.SetFileTypeToBinary()
            writer.Write()
        

class mapPoincare:
    """
    mapPoincare -- Holds the mapped Poincare map together with the mapped streamlines.
    """
    
    def __init__(self, dataDir = 'data', poincareInit = 'poincareInit.vtk', poincare = 'poincare.vtk', dumpFile = 'save.vtk', interpolation = 'weighted', po = []):
        """
        Maps the streamlines from X to x.
        
        call signature:
        
          mapPoincare(dataDir = 'data', poincareInit = 'poincareInit.vtk', poincareFile = 'poincare.vtk', dumpFile = 'save.vtk', interpolation = 'weighted', po = [])
          
        Keyword arguments:
        
         *dataDir*:
            Data directory.
            
         *poincareInit*:
            Read the initial Poincare maps from this file.
        
         *poincareFile*:
            Store the Poincare map in this file.
            
         *dumpFile*:
            Use this file for the mapping x(X,t).
            
         *interpolation*:
            Interpolation of the vector field.
            'mean': takes the mean of the adjacent grid point.
            'weighted': weights the adjacent grid points according to their distance.
            
          *po*:
            The Poincare object for t = 0. If not passed obtain it from 'poincareInit'.
        """
        
        # read the Poincare map at t = 0
        if (po == []):
            po = gm.readPoincare(dataDir = dataDir, poincare = poincareInit) 
        
        # read the current state
        data = gm.readDump(dataDir = dataDir, fileName = dumpFile)
        p = data.p
        xx = data.grid
        
        # mapped tracers
        tracersNew = np.zeros(po.tracers.shape, dtype = po.tracers.dtype)
        # mapped Poincare map
        xNew = np.zeros(po.x.shape)
        yNew = np.zeros(po.y.shape)
        
        # interpolate x(X,t) at S to get x(S,t), where S is the initial streamline
        for j in range(po.x.shape[1]):
            for sl in range(po.sl[j]):
                tracersNew[:,j,sl] = gm.vecInt(po.tracers[:,j,sl], xx, p, interpolation)
            for i in range(po.x.shape[0]):
                if (i == 0):
                    #xyz = np.array([po.x[i,j], po.y[i,j], p.Oz-p.dz])
                    xyz = np.array([po.x[i,j], po.y[i,j], p.Oz])
                else:
                    #xyz = np.array([po.x[i,j], po.y[i,j], p.Oz+p.Lz+p.dz])
                    xyz = np.array([po.x[i,j], po.y[i,j], p.Oz+p.Lz])
                tmp = gm.vecInt(xyz, xx, p, interpolation)
                xNew[i,j] = tmp[0]
                yNew[i,j] = tmp[1]
                
        self.x = xNew
        self.y = yNew
        self.sl = po.sl
        self.tracers = tracersNew
        self.p = p
        self.hMin = po.hMin
        self.hMax = po.hMax
        self.lMax = po.lMax
        self.tol = po.tol
        self.iterMax = po.iterMax
        self.nIter = po.nIter
        self.nSeeds = po.nSeeds
        
        # save into vtk file
        if (poincare != []):
            writer = vtk.vtkPolyDataWriter()
            writer.SetFileName(dataDir + '/' + poincare)
            polyData = vtk.vtkPolyData()
            fieldData = vtk.vtkFieldData()
            # fields containing x and y values
            field = VN.numpy_to_vtk(self.x)
            field.SetName('x')
            fieldData.AddArray(field)
            field = VN.numpy_to_vtk(self.y)
            field.SetName('y')
            fieldData.AddArray(field)
            # field containing length of stream lines for later decomposition
            field = VN.numpy_to_vtk(self.sl)
            field.SetName('sl')
            fieldData.AddArray(field)
            # streamline parameters
            tmp = range(10)            
            tmp[0] = np.array([po.hMin], dtype = 'float32'); field = VN.numpy_to_vtk(tmp[0]); field.SetName('hMin'); fieldData.AddArray(field)
            tmp[1] = np.array([po.hMax], dtype = 'float32'); field = VN.numpy_to_vtk(tmp[1]); field.SetName('hMax'); fieldData.AddArray(field)
            tmp[2] = np.array([po.lMax], dtype = 'float32'); field = VN.numpy_to_vtk(tmp[2]); field.SetName('lMax'); fieldData.AddArray(field)
            tmp[3] = np.array([po.tol], dtype = 'float32'); field = VN.numpy_to_vtk(tmp[3]); field.SetName('tol'); fieldData.AddArray(field)
            tmp[4] = np.array([po.iterMax], dtype = 'int32'); field = VN.numpy_to_vtk(tmp[4]); field.SetName('iterMax'); fieldData.AddArray(field)
            tmp[5] = np.array([po.tol], dtype = 'int32'); field = VN.numpy_to_vtk(tmp[5]); field.SetName('nSeeds'); fieldData.AddArray(field)
            tmp[6] = np.array([po.tol], dtype = 'int32'); field = VN.numpy_to_vtk(tmp[6]); field.SetName('nIter'); fieldData.AddArray(field)
            # fields containing simulation parameters stored in paramFile
            dic = dir(p)
            params = range(len(dic))
            i = 0
            for attr in dic:
                if( attr[0] != '_'):
                    params[i] = getattr(p, attr)
                    params[i] = np.array([params[i]], dtype = type(params[i]))
                    field = VN.numpy_to_vtk(params[i])
                    field.SetName(attr)
                    fieldData.AddArray(field)
                    i += 1
            # all streamlines as continuous array of points
            points = vtk.vtkPoints()
            for q in range(self.x.shape[1]):
                for l in range(self.sl[q]):
                    points.InsertNextPoint(self.tracers[:,q,l])
            polyData.SetPoints(points)
            polyData.SetFieldData(fieldData)
            writer.SetInput(polyData)
            writer.SetFileTypeToBinary()
            writer.Write()
            
            
class readPoincare:
    """
    readPoincare -- Holds the streamlines of the Poincare map at the map.
    """

    def __init__(self, dataDir = 'data', poincare = 'poincareInit.vtk'):
        """
        Read the Poincare streamlines and the map.
        
        call signature:
        
          readStream(dataDir = 'data', poincare = 'poincareInit.vtk')
          
        Keyword arguments:
         *dataDir*:
            Data directory.
            
         *poincare*:
            Read the Poincare streamline and map from this file.
        """
    
        # load the data
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(dataDir + '/' + poincare)
        reader.Update()
        output = reader.GetOutput()
        
        # get the fields
        field = output.GetFieldData()
        nArrays = field.GetNumberOfArrays()
        class params: pass
        p = params()
        for i in range(nArrays):            
            arrayName = field.GetArrayName(i)
            if any(arrayName == np.array(['x', 'y', 'sl'])):
                setattr(self, arrayName, VN.vtk_to_numpy(field.GetArray(arrayName)))
            elif any(arrayName == np.array(['hMin', 'hMax', 'lMax', 'tol', 'iterMax', 'nSeeds', 'nIter'])):
                setattr(self, arrayName, VN.vtk_to_numpy(field.GetArray(arrayName))[0])
            else:
                # change this if parameters can have more than one entry
                setattr(p, arrayName, VN.vtk_to_numpy(field.GetArray(arrayName))[0])
        setattr(self, 'p', p)
        
        # get the points
        points = output.GetPoints()
        pointsData = points.GetData()
        tracers1d = VN.vtk_to_numpy(pointsData)
        tracers1d = np.swapaxes(tracers1d, 0, 1)
        #print tracers1d.shape
        # split into array
        #tracers2d = tracers1d.reshape((3,self.x.shape[1],np.max(self.sl)))
        #tracers2d = np.swapaxes(tracers2d, 1, 2)
        tracers2d = np.zeros([3, self.x.shape[1], np.max(self.sl)])
        #print tracers2d.shape
        sl = 0
        for i in range(len(self.x[0,:])):
            tracers2d[:,i,:self.sl[i]] = tracers1d[:,sl:sl+self.sl[i]]
            sl += self.sl[i]
        setattr(self, 'tracers', tracers2d)
        

def poincareVid(dataDir = 'data', poincareInit = 'poincareInit.vtk', interpolation = 'weighted'): 
    """
    Creates a Poincare map time sequence.
    
    call signature:
    
        poincareVid(dataDir = 'data', poincareInit = 'poincareInit.vtk', interpolation = 'weighted')
    
    Keyword arguments:
    
    *dataDir*:
      Data directory.
        
    *poincareInit*:
        Read the initial Poincare maps from this file.
        
    *interpolation*:
      Interpolation of the vector field.
      'mean': takes the mean of the adjacent grid point.
      'weighted': weights the adjacent grid points according to their distance.
    """
    
    # find the number of dump files and do some sorting
    files = listdir(dataDir)
    files = natsort.natsort(files)
    nFiles = 0
    for i in range(len(files)):
        if (str.find(files[i], 'dump') == 0):
            nFiles += 1
    
    # initialize the array of Poncare objects
    po = []
    
    for f in files:
        if (str.find(f, 'dump') == 0):
            poincareFile = f.replace('dump', 'poincare')
            print(f)
            po.append(gm.mapPoincare(dataDir = dataDir, poincareInit = poincareInit, poincare = poincareFile, dumpFile = f, interpolation = interpolation))
            print(po[-1].p.t)
    
    return po


def readPoincareVid(dataDir = 'data'):
    """
    Reads the Poncare map time sequence.
    
    call signature:
    
        readPoincareVid(dataDir = 'data')
    
    Keyword arguments:
    
    *dataDir*:
      Data directory.
    """
    
    # find the number of dump files and do some sorting
    files = listdir(dataDir)
    files = natsort.natsort(files)
    nFiles = 0
    for i in range(len(files)):
        if ((str.find(files[i], 'poincare') == 0) and (files[i] != 'poincare.vtk') and (str.find(files[i], 'Init') == -1)):
            nFiles += 1
    
    # initialize the array of Poncare objects
    po = []

    for f in files:
        if ((str.find(f, 'poincare') == 0) and (f != 'poincare.vtk') and (str.find(f, 'Init') == -1)):
            print(f)
            po.append(gm.readPoincare(dataDir = dataDir, poincare = f))
    
    return po
