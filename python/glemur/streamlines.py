#
# streamlines.py
#
# Creates and reads streamlines.

import numpy as np
import glemur as gm
import vtk as vtk
from vtk.util import numpy_support as VN
from os import listdir
import natsort

class streamInit:
    """
    streamInit -- Holds the initial streamlines.
    """
    
    def __init__(self, dataDir = 'data', B0File = 'B0.vtk', streamFileInit = 'streamInit.vtk', paramFile = 'save.vtk', interpolation = 'weighted', integration = 'simple', sub = 1, hMin = 2e-3, hMax = 2e4, lMax = 500, tol = 1e-2, iterMax = 1e3, x0 = [], y0 = []):
        """
        Creates, saves and returns the initial streamline map.
        
        call signature:
        
          streamInit(dataDir = 'data', B0File = 'B0.vtk', streamFileInit = 'streamInit.vtk', paramFile = 'save.vtk', interpolation = 'weighted', integration = 'simple', sub = 1, hMin = 2e-3, hMax = 2e2, lMax = 500, tol = 2e-3, iterMax = 1e3, x0 = [], y0 = [])
        
        Trace magnetic streamlines for t = 0 and store the positions in a file.
        
        Keyword arguments:
        
         *dataDir*:
            Data directory.
            
         *B0File*:
            Name of the file storing B(X,0) = B0.
            
         *streamFileInit*:
            Store the streamline in this file. If it is empty [] don't write.
             
         *paramFile*:
            Dump file from where to read the simulation parameters.
             
         *interpolation*:
            Interpolation of the vector field.
            'mean': takes the mean of the adjacent grid point.
            'weighted': weights the adjacent grid points according to their distance.
       
         *integration*:
            Integration method.
            'simple': low order method.
            'RK6': Runge-Kutta 6th order.
       
         *sub*:
            Subsampling of the streamlines.
            
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
        """
        
        data   = gm.readB0(dataDir = dataDir, fileName = B0File)
        #B0     = np.swapaxes(data.bfield, 1, 3)
        B0     = data.bfield
        p      = gm.readParams(dataDir = dataDir, fileName = paramFile)
        self.p = p
        self.p.t = 0
        
        # streamline parameters
        self.sub = sub
        self.hMin = hMin
        self.hMax = hMax
        self.lMax = lMax
        self.tol = tol
        self.iterMax = iterMax
        
        # create initial seeds
        if (x0 == []):
            #x0 = np.arange(p.Ox-p.dx, p.Ox+p.Lx+p.dx, p.dx/sub, dtype = 'float32')
            x0 = np.float32(np.linspace(p.Ox-p.dx, p.Ox+p.Lx+p.dx, np.round(1+(2*p.dx+p.Lx)/p.dx*sub)))
        if (y0 == []):
            #y0 = np.arange(p.Oy-p.dy, p.Oy+p.Ly+p.dy, p.dy/sub, dtype = 'float32')
            y0 = np.float32(np.linspace(p.Oy-p.dy, p.Oy+p.Ly+p.dy, np.round(1+(2*p.dy+p.Ly)/p.dy*sub)))
        self.x0 = x0
        self.y0 = y0
        
        try:
            len(x0)
        except:
            x0 = np.array([x0])
            y0 = np.array([y0])
            
        self.tracers = np.zeros([3, len(x0), len(y0), iterMax], dtype = 'float32')  # tentative streamline length
        self.sl      = np.zeros([len(x0), len(y0)], dtype = 'int32')
        
        tol2 = tol**2
        
        # declare vectors
        xMid    = np.zeros(3)
        xSingle = np.zeros(3)
        xHalf   = np.zeros(3)
        xDouble = np.zeros(3)
        
        # initialize the coefficient for the 6th order adaptive time step RK
        a = np.zeros(6); b = np.zeros((6,5)); c = np.zeros(6); cs = np.zeros(6)
        k = np.zeros((6,3))
        a[1] = 0.2; a[2] = 0.3; a[3] = 0.6; a[4] = 1; a[5] = 0.875
        b[1,0] = 0.2;
        b[2,0] = 3/40.; b[2,1] = 9/40.
        b[3,0] = 0.3; b[3,1] = -0.9; b[3,2] = 1.2
        b[4,0] = -11/54.; b[4,1] = 2.5; b[4,2] = -70/27.; b[4,3] = 35/27.
        b[5,0] = 1631/55296.; b[5,1] = 175/512.; b[5,2] = 575/13824.
        b[5,3] = 44275/110592.; b[5,4] = 253/4096.
        c[0] = 37/378.; c[2] = 250/621.; c[3] = 125/594.; c[5] = 512/1771.
        cs[0] = 2825/27648.; cs[2] = 18575/48384.; cs[3] = 13525/55296.
        cs[4] = 277/14336.; cs[5] = 0.25
    
        # do the streamline tracing
        r = 0
        for u in x0:
            s = 0
            for v in y0:
                # initialize the streamline
                xx = np.array([u, v, p.Oz])
                self.tracers[:,r,s,0] = xx
                dh = np.sqrt(hMax*hMin) # initial step size
                sl = 0
                l = 0
                outside = False
                
                if (integration == 'simple'):
                    #while ((l < lMax) and (xx[2] < p.Oz+p.Lz+p.dz) and (sl < iterMax-1) and (not(np.isnan(xx[0]))) and (outside == False)):
                    while ((l < lMax) and (xx[2] < p.Oz+p.Lz) and (sl < iterMax-1) and (not(np.isnan(xx[0]))) and (outside == False)):
                        # (a) single step (midpoint method)                    
                        xMid = xx + 0.5*dh*gm.vecInt(xx, B0, p, interpolation)
                        xSingle = xx + dh*gm.vecInt(xMid, B0, p, interpolation)
                    
                        # (b) two steps with half stepsize
                        xMid = xx + 0.25*dh*gm.vecInt(xx, B0, p, interpolation)
                        xHalf = xx + 0.5*dh*gm.vecInt(xMid, B0, p, interpolation)
                        xMid = xHalf + 0.25*dh*gm.vecInt(xHalf, B0, p, interpolation)
                        xDouble = xHalf + 0.5*dh*gm.vecInt(xMid, B0, p, interpolation)
                    
                        # (c) check error (difference between methods)
                        dist2 = np.sum((xSingle-xDouble)**2)
                        if (dist2 > tol2):
                            dh = 0.5*dh
                            if (abs(dh) < hMin):
                                print "Error: stepsize underflow"
                                break
                        else:
                            l += np.sqrt(np.sum((xx-xDouble)**2))
                            xx = xDouble.copy()
                            if (abs(dh) < hMin):
                                dh = 2*dh
                            sl += 1
                            self.tracers[:,r,s,sl] = xx.copy()
                            if ((dh > hMax) or (np.isnan(dh))):
                                dh = hMax
                            # check if this point lies outside the domain
                            #if ((xx[0] < p.Ox-p.dx) or (xx[0] > p.Ox+p.Lx+p.dx) or (xx[1] < p.Oy-p.dy) or (xx[1] > p.Oy+p.Ly+p.dy) or (xx[2] < p.Oz-p.dz) or (xx[2] > p.Oz+p.Lz+p.dz)):
                            if ((xx[0] < p.Ox-p.dx) or (xx[0] > p.Ox+p.Lx+p.dx) or (xx[1] < p.Oy-p.dy) or (xx[1] > p.Oy+p.Ly+p.dy) or (xx[2] < p.Oz) or (xx[2] > p.Oz+p.Lz)):
                                outside = True
                                
                if (integration == 'RK6'):
                    #while ((l < lMax) and (xx[2] < p.Oz+p.Lz+p.dz) and (sl < iterMax) and (not(np.isnan(xx[0]))) and (outside == False)):
                    while ((l < lMax) and (xx[2] < p.Oz+p.Lz) and (sl < iterMax) and (not(np.isnan(xx[0]))) and (outside == False)):
                        k[0,:] = dh*gm.vecInt(xx, B0, p, interpolation)                            
                        k[1,:] = dh*gm.vecInt(xx + b[1,0]*k[0,:], B0, p, interpolation)
                        k[2,:] = dh*gm.vecInt(xx + b[2,0]*k[0,:] + b[2,1]*k[1,:], B0, p, interpolation)
                        k[3,:] = dh*gm.vecInt(xx + b[3,0]*k[0,:] + b[3,1]*k[1,:] + b[3,2]*k[2,:], B0, p, interpolation)
                        k[4,:] = dh*gm.vecInt(xx + b[4,0]*k[0,:] + b[4,1]*k[1,:] + b[4,2]*k[2,:] + b[4,3]*k[3,:], B0, p, interpolation)
                        k[5,:] = dh*gm.vecInt(xx + b[5,0]*k[0,:] + b[5,1]*k[1,:] + b[5,2]*k[2,:] + b[5,3]*k[3,:] + b[5,4]*k[4,:], B0, p, interpolation)

                        xNew  = xx + c[0]*k[0,:]  + c[1]*k[1,:]  + c[2]*k[2,:]  + c[3]*k[3,:]  + c[4]*k[4,:]  + c[5]*k[5,:]
                        xNewS = xx + cs[0]*k[0,:] + cs[1]*k[1,:] + cs[2]*k[2,:] + cs[3]*k[3,:] + cs[4]*k[4,:] + cs[5]*k[5,:]

                        delta2 = np.dot((xNew-xNewS), (xNew-xNewS))
                        delta = np.sqrt(delta2)

                        if (delta2 > tol2):
                            dh = dh*(0.9*abs(tol/delta))**0.2
                            if (abs(dh) < hMin):
                                print "Error: step size underflow"
                                break
                        else:
                            l += np.sqrt(np.sum((xx-xNew)**2))
                            xx = xNew                        
                            if (abs(dh) < hMin):
                                dh = 2*dh
                            sl += 1
                            self.tracers[:,r,s,sl] = xx
                            if ((dh > hMax) or (np.isnan(dh))):
                                dh = hMax
                            # check if this point lies outside the domain
                            #if ((xx[0] < p.Ox-p.dx) or (xx[0] > p.Ox+p.Lx+p.dx) or (xx[1] < p.Oy-p.dy) or (xx[1] > p.Oy+p.Ly+p.dy) or (xx[2] < p.Oz-p.dz) or (xx[2] > p.Oz+p.Lz+p.dz)):
                            if ((xx[0] < p.Ox-p.dx) or (xx[0] > p.Ox+p.Lx+p.dx) or (xx[1] < p.Oy-p.dy) or (xx[1] > p.Oy+p.Ly+p.dy) or (xx[2] < p.Oz) or (xx[2] > p.Oz+p.Lz)):
                                outside = True
                        if ((dh > hMax) or (delta == 0) or (np.isnan(dh))):
                            dh = hMax
                    
                self.sl[r,s] = sl+1
                s += 1
            r += 1
        
        # save into vtk file
        if (streamFileInit != []):
            writer = vtk.vtkPolyDataWriter()
            writer.SetFileName(dataDir + '/' + streamFileInit)
            polyData = vtk.vtkPolyData()
            fieldData = vtk.vtkFieldData()
            # fields containing initial x and y values
            field = VN.numpy_to_vtk(self.x0)
            field.SetName('x0')
            fieldData.AddArray(field)
            field = VN.numpy_to_vtk(self.y0)
            field.SetName('y0')
            fieldData.AddArray(field)
            # field containing length of stream lines for later decomposition
            field = VN.numpy_to_vtk(self.sl)
            field.SetName('sl')
            fieldData.AddArray(field)
            # streamline parameters
            tmp = range(10)            
            tmp[0] = np.array([sub], dtype = 'int32'); field = VN.numpy_to_vtk(tmp[0]); field.SetName('sub'); fieldData.AddArray(field)
            tmp[1] = np.array([hMin], dtype = 'float32'); field = VN.numpy_to_vtk(tmp[1]); field.SetName('hMin'); fieldData.AddArray(field)
            tmp[2] = np.array([hMax], dtype = 'float32'); field = VN.numpy_to_vtk(tmp[2]); field.SetName('hMax'); fieldData.AddArray(field)
            tmp[3] = np.array([lMax], dtype = 'float32'); field = VN.numpy_to_vtk(tmp[3]); field.SetName('lMax'); fieldData.AddArray(field)
            tmp[4] = np.array([tol], dtype = 'float32'); field = VN.numpy_to_vtk(tmp[4]); field.SetName('tol'); fieldData.AddArray(field)
            tmp[5] = np.array([iterMax], dtype = 'int32'); field = VN.numpy_to_vtk(tmp[5]); field.SetName('iterMax'); fieldData.AddArray(field)
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
            for r in range(len(x0)):
                for s in range(len(y0)):
                    for sl in range(self.sl[r,s]):
                        points.InsertNextPoint(self.tracers[:,r,s,sl])
            polyData.SetPoints(points)
            polyData.SetFieldData(fieldData)
            writer.SetInput(polyData)
            writer.SetFileTypeToBinary()
            writer.Write()

        
class readStream:
    """
    readStream -- Holds the streamlines.
    """

    def __init__(self, dataDir = 'data', streamFile = 'streamInit.vtk'):
        """
        Read the initial streamlines.
        
        call signature:
        
          readStream(dataDir = 'data', streamFile = 'streamInit.vtk')
          
        Keyword arguments:
         *dataDir*:
            Data directory.
            
         *streamFile*:
            Read the initial streamline from this file.
        """
    
        # load the data
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(dataDir + '/' + streamFile)
        reader.Update()
        output = reader.GetOutput()
        
        # get the fields
        field = output.GetFieldData()
        nArrays = field.GetNumberOfArrays()
        class params: pass
        p = params()
        for i in range(nArrays):            
            arrayName = field.GetArrayName(i)
            if any(arrayName == np.array(['x0', 'y0', 'sl'])):
                setattr(self, arrayName, VN.vtk_to_numpy(field.GetArray(arrayName)))
            elif any(arrayName == np.array(['sub', 'hMin', 'hMax', 'lMax', 'tol', 'iterMax'])):
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
        # split into array
        tracers2d = tracers1d
        tracers2d = np.zeros([3, len(self.x0), len(self.y0), np.max(self.sl)], dtype = tracers1d.dtype)
        sl = 0
        for i in range(len(self.x0)):
            for j in range(len(self.y0)):
                tracers2d[:,i,j,:self.sl[i,j]] = tracers1d[:,sl:sl+self.sl[i,j]]
                sl += self.sl[i,j]
        setattr(self, 'tracers', tracers2d)


class mapStream:
    """
    mapStream -- Holds the mapped streamlines.
    """
    
    def __init__(self, dataDir = 'data', streamFileInit = 'streamInit.vtk', streamFile = 'stream.vtk', dumpFile = 'save.vtk', interpolation = 'weighted', s = []):
        """
        Maps the streamlines from X to x.
        
        call signature:
        
          mapStream(dataDir = 'data', streamFileInit = 'streamInit.vtk', streamFile = 'stream.vtk', dumpFile = 'save.vtk', interpolation = 'weighted')
          
        Keyword arguments:
        
         *dataDir*:
            Data directory.
            
         *streamFileInit*:
            Read the initial streamline in this file.
        
         *streamFile*:
            Store the streamline in this file.
            
         *dumpFile*:
            Use this file for the mapping x(X,t).
            
         *interpolation*:
            Interpolation of the vector field.
            'mean': takes the mean of the adjacent grid point.
            'weighted': weights the adjacent grid points according to their distance.
            
          *s*:
            The streamlines object for t = 0. If not passed obtain it from 'streamFileInit'.
        """
        
        # read the stream lines at t = 0
        if (s == []):
            s = gm.readStream(dataDir = dataDir, streamFile = streamFileInit)            
        #s = gm.readStream(dataDir = dataDir, streamFile = streamFileInit)
        # convert x0 and y0 into array in case of scalars
        try:
            len(s.x0)
        except:
            s.x0 = np.array([s.x0])
            s.y0 = np.array([s.y0])
        
        # read the current state
        data = gm.readDump(dataDir = dataDir, fileName = dumpFile)
        #p = gm.readParams(dataDir = dataDir, fileName = dumpFile)
        p = data.p
        xx = data.grid
        
        # mapped tracers
        tracersNew = np.zeros(s.tracers.shape, dtype = s.tracers.dtype)
        
        # interpolate x(X,t) at S to get x(S,t), where S is the initial streamline
        for i in range(len(s.x0)):
            for j in range(len(s.y0)):
                for sl in range(s.sl[i,j]):
                    tracersNew[:,i,j,sl] = vecInt(s.tracers[:,i,j,sl], xx, p, interpolation)
        self.x0 = s.x0
        self.y0 = s.y0
        self.sl = s.sl
        self.tracers = tracersNew
        self.p = p
        #self.tracers = s.tracers
        self.sub = s.sub
        self.hMin = s.hMin
        self.hMax = s.hMax
        self.lMax = s.lMax
        self.tol = s.tol
        self.iterMax = s.iterMax
        
        # save into vtk file
        if (streamFile != []):
            writer = vtk.vtkPolyDataWriter()
            writer.SetFileName(dataDir + '/' + streamFile)
            polyData = vtk.vtkPolyData()
            fieldData = vtk.vtkFieldData()
            # fields containing initial x and y values
            field = VN.numpy_to_vtk(self.x0)
            field.SetName('x0')
            fieldData.AddArray(field)
            field = VN.numpy_to_vtk(self.y0)
            field.SetName('y0')
            fieldData.AddArray(field)
            # field containing length of stream lines for later decomposition
            field = VN.numpy_to_vtk(self.sl)
            field.SetName('sl')
            fieldData.AddArray(field)
            # streamline parameters
            tmp = range(10)            
            tmp[0] = np.array([s.sub], dtype = 'int32'); field = VN.numpy_to_vtk(tmp[0]); field.SetName('sub'); fieldData.AddArray(field)
            tmp[1] = np.array([s.hMin], dtype = 'float32'); field = VN.numpy_to_vtk(tmp[1]); field.SetName('hMin'); fieldData.AddArray(field)
            tmp[2] = np.array([s.hMax], dtype = 'float32'); field = VN.numpy_to_vtk(tmp[2]); field.SetName('hMax'); fieldData.AddArray(field)
            tmp[3] = np.array([s.lMax], dtype = 'float32'); field = VN.numpy_to_vtk(tmp[3]); field.SetName('lMax'); fieldData.AddArray(field)
            tmp[4] = np.array([s.tol], dtype = 'float32'); field = VN.numpy_to_vtk(tmp[4]); field.SetName('tol'); fieldData.AddArray(field)
            tmp[5] = np.array([s.iterMax], dtype = 'int32'); field = VN.numpy_to_vtk(tmp[5]); field.SetName('iterMax'); fieldData.AddArray(field)
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
            # all stremlines as continuous array of points
            points = vtk.vtkPoints()
            for i in range(len(s.x0)):
                for j in range(len(s.y0)):
                    for k in range(self.sl[i,j]):
                        points.InsertNextPoint(self.tracers[:,i,j,k])
            polyData.SetPoints(points)
            polyData.SetFieldData(fieldData)
            writer.SetInput(polyData)
            writer.SetFileTypeToBinary()
            writer.Write()


def streamVid(dataDir = 'data', streamFileInit = 'streamInit.vtk', interpolation = 'weighted'): 
    """
    Creates a streamline time sequence.
    
    call signature:
    
        streamVid(dataDir = 'data', streamFileInit = 'streamInit.vtk', interpolation = 'weighted')
    
    Keyword arguments:
    
    *dataDir*:
      Data directory.
        
    *streamFileInit*:
      Read the initial streamline in this file.
        
    *interpolation*:
      Interpolation of the vector field.
      'mean': takes the mean of the adjacent grid point.
      'weighted': weights the adjacent grid points according to their distance.
    """
    
    # find the number of dump files
    files = listdir(dataDir)
    files = natsort.natsort(files)
    nFiles = 0
    for i in range(len(files)):
        if (str.find(files[i], 'dump') == 0):
            nFiles += 1
    
    # initialize the array of streamline objects
    s = []
    
    for f in files:
        if (str.find(f, 'dump') == 0):
            streamFile = f.replace('dump', 'stream')
            print f
            s.append(gm.mapStream(dataDir = dataDir, streamFileInit = streamFileInit, streamFile = streamFile, dumpFile = f, interpolation = interpolation))
            print s[-1].p.t
    
    return s


def readStreamVid(dataDir = 'data'):
    """
    Reads the streamline time sequence.
    
    call signature:
    
        readStreamVid(dataDir = 'data')
    
    Keyword arguments:
    
    *dataDir*:
      Data directory.
    """
    
    files = listdir(dataDir)
    files = natsort.natsort(files)
    nFiles = 0
    for i in range(len(files)):
        if ((str.find(files[i], 'stream') == 0) and (files[i] != 'stream.vtk') and (str.find(files[i], 'Init') == -1)):
            nFiles += 1
    
    # initialize the array of streamline objects
    s = []

    for f in files:
        if ((str.find(f, 'stream') == 0) and (f != 'stream.vtk') and (str.find(f, 'Init') == -1)):
            s.append(gm.readStream(dataDir = dataDir, streamFile = f))
    
    return s
  
  
def streamCMapping(s):
    """
    Computes the color mapping of the streamlines from z0 to z1.
    
    call signature::
    
        mapping = streamCMapping(s)
    
    Keyword arguments:
    
    *s*:
      Object returned by streamInit.
    """

    mapping = np.zeros([len(s.x0), len(s.y0), 3])
    
    for i in range(len(s.x0)):
        for j in range(len(s.y0)):
            k = s.sl[i,j]-1
            if ((s.tracers[0,i,j,k] < s.x0[i]) and (s.tracers[1,i,j,k] < s.y0[j])):
                mapping[i,j,:] = [0,1,0]
            if ((s.tracers[0,i,j,k] < s.x0[i]) and (s.tracers[1,i,j,k] >= s.y0[j])):
                mapping[i,j,:] = [1,1,0]
            if ((s.tracers[0,i,j,k] >= s.x0[i]) and (s.tracers[1,i,j,k] < s.y0[j])):
                mapping[i,j,:] = [0,0,1]
            if ((s.tracers[0,i,j,k] >= s.x0[i]) and (s.tracers[1,i,j,k] >= s.y0[j])):
                mapping[i,j,:] = [1,0,0]
                
    return mapping


def vecInt(xx, vv, p, interpolation = 'weighted'):
    """
    Interpolates the field around this position.
    
    call signature:
    
        vecInt(xx, vv, p, interpolation = 'weighted')
    
    Keyword arguments:
    
    *xx*:
      Position vector around which will be interpolated.
    
    *vv*:
      Vector field to be interpolated.
    
    *p*:
      Parameter struct.
    
    *interpolation*:
      Interpolation of the vector field.
      'mean': takes the mean of the adjacent grid point.
      'weighted': weights the adjacent grid points according to their distance.
    """
    
    # find the adjacent indices
    i  = (xx[0]-p.Ox+p.dx)/p.dx
    if (i < 0):
        i = 0
    if (i > p.nx+1):
        i = p.nx+1
    ii = np.array([int(np.floor(i)), \
                    int(np.ceil(i))])
    
    j  = (xx[1]-p.Oy+p.dy)/p.dy    
    if (j < 0):
        j = 0
    if (j > p.ny+1):
        j = p.ny+1
    jj = np.array([int(np.floor(j)), \
                    int(np.ceil(j))])
    
    k  = (xx[2]-p.Oz+p.dz)/p.dz
    if (k < 0):
        k = 0
    if (k > p.nz+1):
        k = p.nz+1
    kk = np.array([int(np.floor(k)), \
                    int(np.ceil(k))])
    
    # interpolate the field
    if (interpolation == 'mean'):
        return np.mean(vv[:,ii[0]:ii[1]+1,jj[0]:jj[1]+1,kk[0]:kk[1]+1], axis = (1,2,3))
    if(interpolation == 'weighted'):
        if (ii[0] == ii[1]): w1 = np.array([1,1])
        else: w1 = (i-ii[::-1])
        if (jj[0] == jj[1]): w2 = np.array([1,1])
        else: w2 = (j-jj[::-1])
        if (kk[0] == kk[1]): w3 = np.array([1,1])
        else: w3 = (k-kk[::-1])            
        weight = abs(w1.reshape((2,1,1))*w2.reshape((1,2,1))*w3.reshape((1,1,2)))
        return np.sum(vv[:,ii[0]:ii[1]+1,jj[0]:jj[1]+1,kk[0]:kk[1]+1]*weight, axis = (1,2,3))/np.sum(weight)
        
