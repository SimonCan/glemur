// global.h

#ifndef GLOBAL_H_
#define GLOBAL_H_

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <arpa/inet.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#define PI             3.141592653589793
#define SQRT2          1.414213562373095

#ifndef PRECISION
    #define PRECISION 32
#endif

#if (PRECISION == 32)
    #define REAL float
    #define REAL_STR "float"
    #define UINT unsigned int
    #define BLCK_EXT 1
#endif
#if (PRECISION == 64)
    #define REAL double
    #define REAL_STR "double"
    #define UINT unsigned long int
    #define BLCK_EXT 0
#endif

// some globally defined variables
extern int  endian;    // tells the code whether machine uses big or little endian

// struct containing all simulation parameters
struct parameters_t {
    int          nx, ny, nz;     // inner mesh points of the domain
    REAL         Lx, Ly, Lz;     // size of the box
    REAL         Ox, Oy, Oz;     // origin
    REAL         dx, dy, dz;     // grid resolution
    REAL         dx1, dy1, dz1;  // inverse grid resolution
    char         bInit[30];         // initial magnetic field
    char         uInit[30];         // initial velocity field
    REAL         ampl;           // amplitude of the initial magnetic field (b0)
    REAL         phi1, phi2;     // first and second twisting angle of the field
    REAL         L1, L2;         // distance of the twist regions from the mid-plane
    REAL         ar, az;         // radius and z-extend of the twist regions
    int          nBlobs;         // number of blobs for the e1, e2 or e3 configuration <= 10
    REAL         width;             // width of flux tubes
    REAL         minor, major;   // minor and major axis of ellipses
    REAL         stretch;        // scaling factor for some configurations
    REAL         bGround;        // background magnetic field strength
    // blob parameters for the e1, e2 and e3 configuration
    REAL         blobXc[10], blobYc[10], blobZc[10], blobZl[10], blobTwist[10], blobScale[10];
    char         initDist[30];   // initial grid distortion profile
    int          initDistCode;   // initial grid distortion profile numerical code. to be used on the GPU
    REAL         initShearA;     // initial grid distortion amplitude
    REAL         initShearB;     // initial grid distortion amplitude
    REAL          initShearK;     // initial grid distortion wave number
    REAL         pert;             // perturbation for the null fan configuration
    REAL         twist;             // twist strength
    bool         fRestart;       // flag, 1 for restarting simulation, 0 for creating new initial state
    int          nCycle;         // total number of time steps
    REAL           dt0;             // initial time step
    REAL         dtMin;             // minimum time interval under which simulations stops
    REAL          maxError;         // maximum allowed error for the time step
    bool         zUpdate;         // flag for updating the z-coordinate for x_{i+1}
    bool         inertia;         // flag for the equation with inertia
    bool         pressure;         // flag for the equation with inertia
    REAL         nu;             // viscosity parameter
    REAL         beta;             // inverse initial density, or weight factor for the pressure gradient
    bool         jxbAver;        // flag for averaging JxB
    REAL         jxbAverWeight;  // weight for the neighbours in case of JxB averaging
    bool         xPeri;             // flag for periodic boundaries in x direction
    bool         yPeri;             // flag for periodic boundaries in y direction
    bool         zPeri;             // flag for periodic boundaries in z direction
    bool         epsilonProf;    // flag for epsilon profile going to 0 at the boundary
    char         jMethod[30];     // method for computing J
    char         pMethod[30];     // method for computing grad(p)
    int          nTs;            // cadence for diagnostic output
    REAL         dtDump;         // time interval for state dumping
    REAL         dtSave;         // time interval for state saving (in case of a crash)
    bool         dumpJ;          // flag set to 1 dumps JJ
    bool         dumpDetJac;     // flag set to 1 dumps detJac
    bool         dumpCellVol;    // flag set to 1 dumps cellVol
    bool         dumpConvexity;  // flag set to 1 dumps convexity
    bool         dumpWedgeMin;   // flag set to 1 dumps the minimum of the wedge product
    bool         redB2;          // flag for computing B2
    bool         redB2f;         // flag for computing the free magnetic energy
    bool         redBMax;        // flag for computing BMax
    bool         redJMax;        // flag for computing JMax
    bool         redJxB_B2Max;   // flag for computing JxB_B2Max
    bool         redEpsilonStar; // force free parameter espilon* (eq, 19, Pontin, Hornig 2009)
    bool         redErrB_1ez;     // error in B-ez, or deviation from a straight field
    bool         redErrXb_XbAn;     // error/deviation from analytically computed relaxed distortion
    bool         redConvex;         // monitors the convexity of the system (1 = convex, 0 = concave)
    bool         redWedgeMin;     // monitors the minimum of the 8 wedge products
    bool         redU2;          // flag for computing U2
    bool         redUMax;        // flag for computing UMax
};

// struct containing the reduction results
struct red_t {
    REAL  B2;              // total magnetic energy
    REAL  B2res;          // residual magnetic energy
    REAL  BMax;              // maximum value for |B|
    REAL  JMax;              // maximum value for |J|
    REAL  JxB_B2Max;      // maximum value for JxB/B**2
    REAL  epsilonStar;    // force free parameter espilon* (eq, 19, Pontin, Hornig 2009)
    REAL  errB_1ez;          // error in B-ez, or deviation from a straight field
    REAL  errXb_XbAn;      // error/deviation from analytically computed relaxed distortion
    REAL  convex;         // = 1 for all grid points convex, 0 otherwise
    REAL  wedgeMin;          // minimum of the 8 wedge products
    REAL  U2;              // total kinetic energy
    REAL  UMax;              // maximum value for |U|
};

// struct containing large host arrays
struct varsHost_t {
    // host memory
    REAL                *B0;              // initial magnetic field
    REAL                *B;               // current magnetic field
    REAL                *JJ;              // electric current density
    REAL                *xb;              // deformed grid
    REAL                *uu;            // velocity field
    REAL                *detJac;          // determinant of the Jacobian for the transformation
    REAL                *cellVol;         // volume element in positive xyz direction from grid point
    REAL                *convexity;       // the grid's "convexity"
    REAL                  *wedgeMin;        // minimum of the 8 wedge products at each grid point
};

struct varsDev_t {
    // pointers to the global device memory
    REAL    *xb, *B0;       // deformed grid and initial magnetic field on the device
    REAL    *uu;            // velocity field
    REAL    *B;             // current magnetic field used to communicate boundaries
    REAL    *J;             // electric current density
    REAL    *gradP;            // gradient of the pressure
    REAL    *kk;            // vectors k for the adaptive time step Runge Kutta method
    REAL    *xb_new;        // new deformed grid
    REAL    *maxDelta;      // maximum for Delta (error) in the time stepping method
    REAL    *xb_tmp;        // temporary storage for xb to counteract racing issues
    REAL    *uu_new;        // new deformed grid
    REAL    *uu_tmp;        // new deformed grid
    REAL    *Jmag;            // magnitude of the current
    REAL    *JxB_B2;        // norm of the cross product of J and B divided by B**2
    REAL    *JB_B2;            // dot product of J and B divided by B**2
    REAL    *epsStar;        // force free parameter
    REAL    *detJac;        // determinant of the Jacobian for the transformation
    REAL    *cellVol;       // volume element in positive xyz direction from grid point
    REAL    *convexity;     // the grid's "convexity"
    REAL      *wedgeMin;        // minimum of the 8 wedge products at each grid point
    REAL    *B_1ez2;        // error of B-ez
    REAL    *xb_xbAn;        // error of xb-xbAn (difference between numerical and analytical xb)
    REAL    *B2_det;        // energy density B**2*detJac
    REAL    *U2_det;        // energy density U**2*detJac
    REAL    *Umag;            // magnitude of the velocity
};

// constants on the device for fast access
extern __constant__ struct parameters_t dev_p; // simulation parameters on the device
extern __constant__ REAL   a_i[6];             // coefficients a_i for the adaptive step size RG
extern __constant__ REAL   b_ij[7*6];          // coefficients b_ij for the adaptive step size RG
extern __constant__ REAL   c_i[6];             // coefficients c_i for the adaptive step size RG
extern __constant__ REAL   c_star_i[6];        // coefficients c*_i for the adaptive step size RG

#endif /* ODEMI_H_ */
