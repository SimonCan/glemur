#
# maths.py
#
# Contains some basic mathematical operations.

import numpy as np
import vtk as vtk
#from vtk import vtkStructuredPointsReader
#from vtk.util import numpy_support as VN


class grad:
    """
    grad -- Holds gradient of the scalar field.
    """
    
    def __init__(self, field, grid, dx, dy, dz):
        """
        Computes the gradient of a scalar field.
        
        call signature:
        
          grad(field, grid, dx, dy, dz)
          
        Keyword arguments:
        
         *field*:
            The array with the 3d scalar field.
            
         *grid*:
            Name of the file storing B(X,0) = B0.
         
         *dx, dy, dz*
            The initial grid spacing.
        """

        # Compute the Jacobian for the mapping.
        jac = np.zeros((grid.shape[0], grid.shape[0], grid.shape[1]-2, grid.shape[2]-2, grid.shape[3]-2))
        jac[0, ...] = (grid[:, 2:, 1:-1, 1:-1] - grid[:, :-2, 1:-1, 1:-1])/dx/2
        jac[1, ...] = (grid[:, 1:-1, 2:, 1:-1] - grid[:, 1:-1, :-2, 1:-1])/dy/2
        jac[2, ...] = (grid[:, 1:-1, 1:-1, 2:] - grid[:, 1:-1, 1:-1, :-2])/dz/2
        # Compute the determinand of the Jacobian.
        detJac = jac[0, 0]*jac[1, 1]*jac[2, 2] + jac[0, 1]*jac[1, 2]*jac[2, 0] + jac[0, 2]*jac[1, 0]*jac[2, 1] - \
                 jac[0, 0]*jac[1, 2]*jac[2, 1] - jac[0, 1]*jac[1, 0]*jac[2, 2] - jac[0, 2]*jac[1, 1]*jac[2, 0]                 
        # Compute the inverse Jacobian for the mapping.
        jacInv = np.zeros(jac.shape)
        jacInv[0, 0, ...] = jac[1, 1, ...]*jac[2, 2, ...] - jac[1, 2, ...]*jac[2, 1, ...]
        jacInv[0, 1, ...] = jac[0, 2, ...]*jac[2, 1, ...] - jac[0, 1, ...]*jac[2, 2, ...]
        jacInv[0, 2, ...] = jac[0, 1, ...]*jac[1, 2, ...] - jac[0, 2, ...]*jac[1, 1, ...]
        jacInv[1, 0, ...] = jac[1, 2, ...]*jac[2, 0, ...] - jac[1, 0, ...]*jac[2, 2, ...]
        jacInv[1, 1, ...] = jac[0, 0, ...]*jac[2, 2, ...] - jac[0, 2, ...]*jac[2, 0, ...]
        jacInv[1, 2, ...] = jac[0, 2, ...]*jac[1, 0, ...] - jac[0, 0, ...]*jac[1, 2, ...]        
        jacInv[2, 0, ...] = jac[1, 0, ...]*jac[2, 1, ...] - jac[1, 1, ...]*jac[2, 0, ...]
        jacInv[2, 1, ...] = jac[0, 1, ...]*jac[2, 0, ...] - jac[0, 0, ...]*jac[2, 1, ...]
        jacInv[2, 2, ...] = jac[0, 0, ...]*jac[1, 1, ...] - jac[0, 1, ...]*jac[1, 0, ...]
        jacInv = jacInv/detJac
        
        # Compute the gradient of the field in the X-coordinate system.
        gradFx = np.zeros(jac.shape[1:])
        gradFx[0, ...] = (field[2:, 1:-1, 1:-1] - field[:-2, 1:-1, 1:-1])/dx/2
        gradFx[1, ...] = (field[1:-1, 2:, 1:-1] - field[1:-1, :-2, 1:-1])/dy/2
        gradFx[2, ...] = (field[1:-1, 1:-1, 2:] - field[1:-1, 1:-1, :-2])/dz/2
        # Transform this into the new coordinates.
        gradFy = np.zeros(jac.shape[1:])
        gradFy[0, ...] = np.sum(jacInv[0, :, ...]*gradFx[:, ...], axis=0)
        gradFy[1, ...] = np.sum(jacInv[1, :, ...]*gradFx[:, ...], axis=0)
        gradFy[2, ...] = np.sum(jacInv[2, :, ...]*gradFx[:, ...], axis=0)

        setattr(self, 'gradF', gradFy)
        
