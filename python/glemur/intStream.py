#
# intStream.py
#
# Integrates quantities along streamlines.

import numpy as np
import glemur as gm
import vtk as vtk
from vtk.util import numpy_support as VN
from os import listdir
import natsort


class intStream:
    """
    intStream -- Holds the field line integrated quantities.
    """

    def __init__(self, intQ = ['Jp'], dataDir = 'data', streamFileInit = 'streamInit.vtk', streamFile = 'stream.vtk', dumpFile = 'save.vtk', mapFile = 'map.vtk', interpolation = 'weighted', s0 = [], s = []):
        """
        Compute the integrated quantities. If the streamline objects are not passed, read them.
        Make sure 'streamFile' and 'dumpFile' are from the same time or you will get unexpected results.
        
        call signature:
        
          intStreamintQ = ['Jp'], dataDir = 'data', streamFileInit = 'streamInit.vtk', streamFile = 'stream.vtk', dumpFile = 'save.vtk', mapFile = 'map.vtk', interpolation = 'weighted', s0 = [], s = [])
          
        Keyword arguments:
        
         *intQ*:
           Quantity to be integrated along the streamlines.
           Options: 'Jp', 'JB', 'lam', 'lamS', 'deltaLam', 'epsilon', 'epsilonS', 'l', 'twist', 'mu'
           
         *dataDir*:
            Data directory.
            
         *streamFileInit*:
            Read the initial streamline at t = 0 from this file.
            
         *streamFile*:
            Read the streamline at t from this file.
            
         *dumpFile:
            File containing the physical values.
            
         *mapFile*:
            Store the mapping here.
            
         *interpolation*:
            Interpolation of the vector field.
            'mean': takes the mean of the adjacent grid point.
            'weighted': weights the adjacent grid points according to their distance.
            
          *s0*:
            The streamlines object for t = 0. If not passed obtain it from 'streamFileInit'.
            
          *s*:
            The streamlines object at t. If not passed obtain it from 'streamFile'.
        """

        if (s0 == []):
            s0 = gm.readStream(dataDir = dataDir, streamFile = streamFileInit)
        if (s == []):
            s = gm.readStream(dataDir = dataDir, streamFile = streamFile)
        data = gm.readDump(dataDir = dataDir, fileName = dumpFile)
        p    = s.p
        
        self.x0 = s.x0
        self.y0 = s.y0
        self.tracers0 = s.tracers[:,:,:,0]  
        self.p  = p
        self.sub = s.sub
        self.hMin = s.hMin
        self.hMax = s.hMax
        self.lMax = s.lMax
        self.tol = s.tol
        self.iterMax = s.iterMax
        
        # vtk object containing the grid
        points = vtk.vtkPoints()
        xx = s.tracers[:,:,:,0]
        xx = xx[:,:,:,np.newaxis]
        for i in range(len(s.x0)):
            for j in range(len(s.y0)):
                points.InsertNextPoint(xx[:,j,i,0])
        grid = vtk.vtkStructuredGrid()
        grid.SetDimensions([len(s.y0),len(s.x0),1])
        grid.SetPoints(points)
        
        # make intQ iterable even if it contains only one element
        if (isinstance(intQ, str)):
            intQ = [intQ]
        intQ = np.array(intQ)
            
        # check if we need to J
        needJ = np.any(np.array(['Jp', 'JB', 'lam', 'lamS', 'deltaLam', 'epsilon', 'epsilonS', 'twist', 'mu']) == intQ.reshape((intQ.shape[0],1)))        
        # check if we need B
        needB = np.any(np.array(['JB', 'lam', 'lamS', 'deltaLam', 'epsilon', 'epsilonS', 'twist', 'mu']) == intQ.reshape((intQ.shape[0],1)))
        # check if we need the length dl
        needDl = np.any(np.array(['JB', 'lam', 'lamS', 'deltaLam', 'epsilon', 'epsilonS', 'l', 'twist', 'mu']) == intQ.reshape((intQ.shape[0],1)))
        
        # check if the data has the necessarry attributes
        if (needJ and not(hasattr(data, 'jfield'))):
            print "Error: missing attribute 'jfield' in data."
            return -1
        if (needJ and not(hasattr(data, 'bfield'))):
            print "Error: missing attribute 'bfield' in data."
            return -1

        # add integrated quantities to intQ (because we can)
        addQ = set([])
        if needJ:
            addQ = addQ.union(set(['Jp', 'JB', 'lam', 'lamS', 'deltaLam', 'epsilon', 'epsilonS', 'twist', 'mu']))
        if needB:
            addQ = addQ.union(set(['JB', 'lam', 'lamS', 'deltaLam', 'epsilon', 'epsilonS', 'twist', 'mu']))
        if needDl:
            addQ = addQ.union(set(['JB', 'lam', 'lamS', 'deltaLam', 'epsilon', 'epsilonS', 'l', 'twist', 'mu']))
        intQ = np.array(list(addQ))
        
        # create arrays for the vtk i/o
        arrays = {}
        for q in intQ:
            arr = vtk.vtkFloatArray()
            arr.SetName(q)
            arr.SetNumberOfComponents(1)
            arr.SetNumberOfTuples(grid.GetNumberOfPoints())            
            arrays[q] = arr
         
        # initialize all arrays
        print s.tracers.shape[1:3]
        if needJ:
            Jp = np.zeros(s.tracers.shape[1:3])
        if (needJ and needB):
            JB = np.zeros(s.tracers.shape[1:3])
            lam = np.zeros(s.tracers.shape[1:3])
            lamS = np.zeros(s.tracers.shape[1:3])
            deltaLam = np.zeros(s.tracers.shape[1:3])
            epsilon = np.zeros(s.tracers.shape[1:3])
            epsilonS = np.zeros(s.tracers.shape[1:3])
            twist = np.zeros(s.tracers.shape[1:3])
            mu = np.zeros(s.tracers.shape[1:3])
            Jn = np.zeros(3)    # normalized current
        if needDl:
            l2d = np.zeros(s.tracers.shape[1:3])
            
        for i in range(len(s.x0)):
            for j in range(len(s.y0)):
                xx1 = s.tracers[:,i,j,0]
                if needJ:
                    JJ1 = gm.vecInt(s0.tracers[:,i,j,0], data.jfield, p, interpolation = interpolation)
                if needB:
                    BB1 = gm.vecInt(s0.tracers[:,i,j,0], data.bfield, p, interpolation = interpolation)
                    lamTmp = np.zeros(s.sl[i,j])
                    lamTmp[0] = np.dot(JJ1, BB1)/np.dot(BB1, BB1)
                    epsilonTmp = np.zeros(s.sl[i,j])
                    epsilonTmp[0] = np.linalg.norm(np.cross(JJ1, BB1))/np.dot(BB1, BB1)
                if needDl:
                    l = 0
                    ll = np.zeros(s.sl[i,j])
                    ll[0] = l
                for k in range(1, s.sl[i,j]):
                    xx2 = s.tracers[:,i,j,k]
                    if needJ:
                        JJ2 = gm.vecInt(s0.tracers[:,i,j,k], data.jfield, p, interpolation = interpolation)
                    if needB:
                        BB2 = gm.vecInt(s0.tracers[:,i,j,k], data.bfield, p, interpolation = interpolation)
                    if needDl:
                        dl = np.linalg.norm(xx2-xx1)
                        l += dl
                        ll[k-1] = l
                    # integrate the quantities
                    if needJ:
                        Jp[i,j] += np.dot((JJ2+JJ1), (xx2-xx1)) # division by 2 done later
                    if (needJ and needB):
                        JB[i,j] += np.dot((JJ2+JJ1), (BB2+BB1))*dl   # division by 4 done later
                        #lamTmp[k-1] = np.dot((JJ2+JJ1), (BB2+BB1))/np.dot((BB2+BB1), (BB2+BB1))
                        lamTmp[k] = np.dot(JJ2, BB2)/np.dot(BB2, BB2)
                        lam[i,j] += (lamTmp[k]+lamTmp[k-1])*dl/2
                        deltaLam[i,j] += (np.dot(JJ2,BB2)/np.dot(BB2,BB2) - np.dot(JJ1,BB1)/np.dot(BB1,BB1))/dl
                        #epsilonTmp[k-1] = np.linalg.norm(np.cross((JJ2+JJ1), (BB2+BB1)))/np.dot((BB2+BB1), (BB2+BB1))
                        epsilonTmp[k] = np.linalg.norm(np.cross(JJ2, BB2))/np.dot(BB2, BB2)
                        epsilon[i,j] += epsilonTmp[k]*dl
                        #twist[i,j] += np.dot((JJ2+JJ1), (BB2+BB1))/(np.linalg.norm(JJ2+JJ1)*np.linalg.norm(BB2+BB1))*dl
                        # take the norm such that for small vectors errors are small
                        Jtmp = (JJ2+JJ1)
                        Jn[0] = np.sign(Jtmp[0])/np.sqrt(1 + (Jtmp[1]/Jtmp[0])**2 + (Jtmp[2]/Jtmp[0])**2)
                        Jn[1] = np.sign(Jtmp[1])/np.sqrt(1 + (Jtmp[2]/Jtmp[1])**2 + (Jtmp[0]/Jtmp[1])**2)
                        Jn[2] = np.sign(Jtmp[2])/np.sqrt(1 + (Jtmp[0]/Jtmp[2])**2 + (Jtmp[1]/Jtmp[2])**2)
                        Jn = Jn/np.linalg.norm(Jn)
                        twist[i,j] += np.dot(Jn, (BB2+BB1))/np.linalg.norm(BB2+BB1)*dl
                    xx1 = xx2; JJ1 = JJ2; BB1 = BB2
                if needJ:
                    Jp[i,j] = Jp[i,j]/2
                    arrays['Jp'].SetValue(i+j*len(s.x0), Jp[i,j])
                if (needJ and needB):
                    JB = JB/4
                    lam[i,j] = lam[i,j]/l
                    
                    dLamTmp = (lamTmp[1:] - lamTmp[:-1]) / (ll[1:] - ll[:-1])
                    dLamTmp[np.isnan(dLamTmp)] = 0
                    lamS[i,j] = np.max(dLamTmp)
                    
                    epsilon[i,j] = epsilon[i,j]/l
                    epsilonTmp = (epsilonTmp[1:] - epsilonTmp[:-1]) / (ll[1:] - ll[:-1])
                    epsilonTmp[np.isnan(epsilonTmp)] = 0
                    epsilonS[i,j] = np.max(epsilonTmp)
                    
                    twist[i,j] = twist[i,j]/l
                    if (ll.shape[0] > 1):
                        idx = np.where(dLamTmp == lamS[i,j])[0][0]
                        #print idx, mu.shape, lamS.shape, ll.shape, lamTmp.shape
                        mu[i,j] = lamS[i,j]*2*(ll[idx+1] - ll[idx])/(lamTmp[idx+1] + lamTmp[idx])
                    else:
                        mu[i,j] = lamS[i,j]*2*ll[idx]/lamTmp[idx]
                        
                    arrays['JB'].SetValue(i+j*len(s.x0), JB[i,j])
                    arrays['lam'].SetValue(i+j*len(s.x0), lam[i,j])
                    arrays['lamS'].SetValue(i+j*len(s.x0), lamS[i,j])
                    arrays['deltaLam'].SetValue(i+j*len(s.x0), deltaLam[i,j])
                    arrays['epsilon'].SetValue(i+j*len(s.x0), epsilon[i,j])
                    arrays['epsilonS'].SetValue(i+j*len(s.x0), epsilonS[i,j])
                    arrays['twist'].SetValue(i+j*len(s.x0), twist[i,j])
                    arrays['mu'].SetValue(i+j*len(s.x0), mu[i,j])
                if needDl:
                    l2d[i,j] = l
                    arrays['l'].SetValue(i+j*len(s.x0), l2d[i,j])
        if needJ:
            self.Jp = Jp
        if (needJ and needB):
            self.JB = JB
            self.lam = lam
            self.lamS = lamS
            self.deltaLam = deltaLam
            self.epsilon = epsilon
            self.epsilonS = epsilonS
            self.twist = twist
            self.mu = mu
        if needDl:
            self.l = l2d
        # add results to the vtk array
        for q in intQ:
            grid.GetPointData().AddArray(arrays[q])
       
        fieldData = vtk.vtkFieldData()
        
        # save parameter x0, y0        
        field = VN.numpy_to_vtk(self.x0)
        field.SetName('x0')
        fieldData.AddArray(field)
        field = VN.numpy_to_vtk(self.y0)
        field.SetName('y0')
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
                
        grid.SetFieldData(fieldData)
        
        # write into vtk file
        writer = vtk.vtkStructuredGridWriter()
        writer.SetFileName(dataDir + '/' + mapFile)
        writer.SetInput(grid)
        writer.SetFileTypeToBinary()
        writer.Write()


class readMap:
    """
    readMap -- Holds the integration maps.
    """

    def __init__(self, dataDir = 'data', mapFile = 'map.vtk'):
        """
        Read the initial streamlines.
        
        call signature:
        
          readStream(dataDir = 'data', mapFile = 'map.vtk')
          
        Keyword arguments:
         *dataDir*:
            Data directory.
            
         *mapFile*:
            Read the mapping from this file.
        """
    
        # load the data
        reader = vtk.vtkStructuredGridReader()
        reader.SetFileName(dataDir + '/' + mapFile)
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
        
        points = output.GetPoints()
        data = points.GetData()
        tracers0 = VN.vtk_to_numpy(data)
        tracers0 = np.reshape(tracers0, (p.ny+2,p.nx+2,3))
        tracers0 = np.swapaxes(tracers0, 0, 2)
        self.tracers0 = tracers0
        
        # get the point data
        pointData = output.GetPointData()
        nArrays = pointData.GetNumberOfArrays()
        for i in range(nArrays):
            name = pointData.GetArrayName(i)
            array = VN.vtk_to_numpy(pointData.GetArray(i))
            array = np.reshape(array, (p.nx+2, p.ny+2))
            array = np.swapaxes(array, 0, 1)
            setattr(self, name, array)
            

def mapVid(intQ = ['Jp'], dataDir = 'data', streamFileInit = 'streamInit.vtk', interpolation = 'weighted'): 
    """
    Creates a integration map time sequence.
    
    call signature:
    
        mapVid(intQ = ['Jp'], dataDir = 'data', streamFileInit = 'streamInit.vtk', interpolation = 'weighted')
    
    Keyword arguments:
    
    *intQ*:
        Quantity to be integrated along the streamlines.
           
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
    m = []
    
    for f in files:
        if (str.find(f, 'dump') == 0):
            streamFile = f.replace('dump', 'stream')
            mapFile = f.replace('dump', 'map')
            print f
            m.append(gm.intStream(intQ = intQ, dataDir = dataDir, streamFileInit = streamFileInit, streamFile = streamFile, dumpFile = f, mapFile = mapFile, interpolation = interpolation))
            print m[-1].p.t
    
    return m


def readMapVid(dataDir = 'data'):
    """
    Reads the integration map time sequence.
    
    call signature:
    
        readMapVid(dataDir = 'data')
    
    Keyword arguments:
    
    *dataDir*:
      Data directory.
    """
    
    files = listdir(dataDir)
    files = natsort.natsort(files)
    nFiles = 0
    for i in range(len(files)):
        if ((str.find(files[i], 'map') == 0) and (files[i] != 'map.vtk') and (str.find(files[i], 'Init') == -1)):
            nFiles += 1
    
    # initialize the array of streamline objects
    m = []

    for f in files:
        if ((str.find(f, 'map') == 0) and (f != 'map.vtk') and (str.find(f, 'Init') == -1)):
            m.append(gm.readMap(dataDir = dataDir, mapFile = f))
    
    return m

    
    