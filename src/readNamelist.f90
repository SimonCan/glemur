!
! Read the simulation parameters from 'params.in'.
!

#ifndef PRECISION
#define PRECISION 32
#endif

#define SQRT2          1.414213562373095

#if (PRECISION == 32)
#define CREAL c_float
#endif
#if (PRECISION == 64)
#define CREAL c_double
#endif

subroutine readNamelist(p) bind(c, name='readnamelist')

    use, intrinsic :: iso_c_binding, only:c_float, c_double, c_int, c_bool, c_char, C_NULL_CHAR
    implicit none

    ! NB: the order has to be the same as in the C declaration in global.h
    type, bind(c) :: parameters_t
        integer(kind=c_int)     :: nx, ny, nz
        real(kind=CREAL)      :: Lx, Ly, Lz
        real(kind=CREAL)      :: Ox, Oy, Oz
        real(kind=CREAL)      :: dx, dy, dz
        real(kind=CREAL)      :: dx1, dy1, dz1
        character(kind=c_char, len=30) :: bInit
        character(kind=c_char, len=30) :: uInit
        real(kind=CREAL)      :: ampl
        real(kind=CREAL)      :: phi1, phi2
        real(kind=CREAL)      :: L1, L2
        real(kind=CREAL)      :: ar, az
        integer(kind=c_int)   :: nBlobs
        real(kind=CREAL)      :: width
        real(kind=CREAL)      :: minor, major
        real(kind=CREAL)      :: stretch
        real(kind=CREAL)      :: bGround
        real(kind=CREAL), dimension(10) :: blobXc, blobYc, blobZc, blobZl, blobTwist, blobScale
        character(kind=c_char, len=30) ::  initDist
        integer(kind=c_int)   :: initDistCode
        real(kind=CREAL)      :: initShearA
        real(kind=CREAL)      :: initShearB
        real(kind=CREAL)      :: initShearK
        real(kind=CREAL)      :: pert
        real(kind=CREAL)      :: twist
        logical(kind=c_bool)  :: fRestart
        integer(kind=c_int)   :: nCycle
        real(kind=CREAL)      :: dt0
        real(kind=CREAL)      :: dtMin
        real(kind=CREAL)      :: maxError
        logical(kind=c_bool)  :: zUpdate
        logical(kind=c_bool)  :: inertia
        logical(kind=c_bool)  :: pressure
        real(kind=CREAL)      :: nu
        real(kind=CREAL)      :: beta
        logical(kind=c_bool)  :: jxbAver
        real(kind=CREAL)      :: jxbAverWeight
        logical(kind=c_bool)  :: xPeri
        logical(kind=c_bool)  :: yPeri
        logical(kind=c_bool)  :: zPeri
        logical(kind=c_bool)  :: epsilonProf
        character(kind=c_char, len=30) :: jMethod
        character(kind=c_char, len=30) :: pMethod
        integer(kind=c_int)   :: nTs
        real(kind=CREAL)      :: dtDump
        real(kind=CREAL)      :: dtSave
        logical(kind=c_bool)  :: dumpJ
        logical(kind=c_bool)  :: dumpDetJac
        logical(kind=c_bool)  :: dumpCellVol
        logical(kind=c_bool)  :: dumpConvexity
        logical(kind=c_bool)  :: dumpWedgeMin
        logical(kind=c_bool)  :: redB2
        logical(kind=c_bool)  :: redB2f
        logical(kind=c_bool)  :: redBMax ! not implemented yet
        logical(kind=c_bool)  :: redJMax
        logical(kind=c_bool)  :: redJxB_B2Max
        logical(kind=c_bool)  :: redEpsilonStar
        logical(kind=c_bool)  :: redErrB_1ez
        logical(kind=c_bool)  :: redErrXb_XbAn
        logical(kind=c_bool)  :: redConvex
        logical(kind=c_bool)  :: redWedgeMin
        logical(kind=c_bool)  :: redU2
        logical(kind=c_bool)  :: redUMax
    end type

    type(parameters_t), intent(inout) :: p

    ! these are used for reading the params.in file
    integer(kind=c_int)   :: nx = 16, ny = 16, nz = 16
    real(kind=CREAL)      :: Lx = 8., Ly = 8., Lz = 20.
    real(kind=CREAL)      :: Ox = 0., Oy = 0., Oz = 0.
    character(kind=c_char, len=30) :: bInit = "Pontin09"
    character(kind=c_char, len=30) :: uInit = "nil"
    real(kind=CREAL)      :: ampl = 1.
    real(kind=CREAL)      :: phi1 = 2., phi2 = -2.
    real(kind=CREAL)      :: L1 = -4., L2 = 4.
    real(kind=CREAL)      :: ar = SQRT2, az = 2.
    integer(kind=c_int)   :: nBlobs = 0
    real(kind=CREAL)      :: width = 0.6
    real(kind=CREAL)      :: minor = 1.0, major = 2.5
    real(kind=CREAL)      :: stretch = 1.0
    real(kind=CREAL)      :: bGround = 0.
    real(kind=CREAL), dimension(10) :: blobXc, blobYc, blobZc, blobZl, blobTwist, blobScale ! warning: no default values
    character(kind=c_char, len=30) ::  initDist = "none"
    real(kind=CREAL)      :: initShearA = 0.
    real(kind=CREAL)      :: initShearB = 0.
    real(kind=CREAL)      :: initShearK = 1.
    real(kind=CREAL)      :: pert = 1.
    real(kind=CREAL)      :: twist = 1.
    logical(kind=c_bool)  :: fRestart = .False.
    integer(kind=c_int)   :: nCycle = 10
    real(kind=CREAL)      :: dt0 = 3e-4
    real(kind=CREAL)      :: dtMin = 1e-9
    real(kind=CREAL)      :: maxError = 5e-5
    logical(kind=c_bool)  :: zUpdate = .True.
    logical(kind=c_bool)  :: inertia = .False.
    logical(kind=c_bool)  :: pressure = .False.
    real(kind=CREAL)      :: nu = 1
    real(kind=CREAL)      :: beta = 1
    logical(kind=c_bool)  :: jxbAver = .False.
    real(kind=CREAL)      :: jxbAverWeight = 0.
    logical(kind=c_bool)  :: xPeri = .False.
    logical(kind=c_bool)  :: yPeri = .False.
    logical(kind=c_bool)  :: zPeri = .False.
    logical(kind=c_bool)  :: epsilonProf = .True.
    character(kind=c_char, len=30) :: jMethod = "Stokes"
    character(kind=c_char, len=30) :: pMethod = "Classic"
    integer(kind=c_int)   :: nTs = 10
    real(kind=CREAL)      :: dtDump = 10.
    real(kind=CREAL)      :: dtSave = 1.
    logical(kind=c_bool)  :: dumpJ = .False.
    logical(kind=c_bool)  :: dumpDetJac = .False.
    logical(kind=c_bool)  :: dumpCellVol = .False.
    logical(kind=c_bool)  :: dumpConvexity = .False.
    logical(kind=c_bool)  :: dumpWedgeMin = .False.
    logical(kind=c_bool)  :: redB2 = .False.
    logical(kind=c_bool)  :: redB2f = .False.
    logical(kind=c_bool)  :: redBMax = .False. ! not implemented yet
    logical(kind=c_bool)  :: redJMax = .False.
    logical(kind=c_bool)  :: redJxB_B2Max = .False.
    logical(kind=c_bool)  :: redEpsilonStar = .False.
    logical(kind=c_bool)  :: redErrB_1ez = .False.
    logical(kind=c_bool)  :: redErrXb_XbAn = .False.
    logical(kind=c_bool)  :: redConvex = .False.
    logical(kind=c_bool)  :: redWedgeMin = .False.
    logical(kind=c_bool)  :: redU2 = .False.
    logical(kind=c_bool)  :: redUMax = .False.

    namelist /comp/ nx, ny, nz

    namelist /start/ Lx, Ly, Lz, &
        Ox, Oy, Oz, &
        bInit, uInit, &
        ampl, &
        phi1, phi2, &
        L1, L2, &
        ar, az, &
        nBlobs, blobXc, blobYc, blobZc, blobZl, blobTwist, blobScale, &
        width, minor, major, stretch, bGround, &
        initDist, initShearA, initShearB, initShearK, pert, twist, &
        fRestart

    namelist /run/ nCycle, dt0, dtMin, maxError, zUpdate, inertia, pressure, nu, beta, jxbAver, jxbAverWeight, &
        xPeri, yPeri, zPeri, epsilonProf, jMethod, pMethod

    namelist /io/ nTs, dtDump, dtSave, dumpJ, dumpDetJac, dumpCellVol, dumpConvexity, dumpWedgeMin, &
        redB2, redB2f, redBMax, redJMax, redJxB_B2Max, redEpsilonStar, redErrB_1ez, redErrXb_XbAn, &
        redConvex, redWedgeMin, redU2, redUMax

    open(unit=1, file='params.in', status='old')
    read(unit=1, nml=comp)
    read(unit=1, nml=start)
    read(unit=1, nml=run)
    read(unit=1, nml=io)
    close(unit=1)

    ! copy values into struct (is there a more direct and cleaner method?)
    p%nx = nx; p%ny = ny; p%nz = nz

    p%Lx = Lx;        p%Ly = Ly;        p%Lz = Lz
    p%Ox = Ox;        p%Oy = Oy;        p%Oz = Oz
    p%dx = Lx/(nx-1); p%dy = Ly/(ny-1); p%dz = Lz/(nz-1);  ! without boundary layer
    p%dx1 = 1/p%dx;   p%dy1 = 1/p%dy;   p%dz1 = 1/p%dz
    p%bInit    = bInit//C_NULL_CHAR
    p%uInit    = uInit//C_NULL_CHAR
    p%ampl     = ampl
    p%phi1     = phi1
    p%phi2     = phi2
    p%L1       = L1
    p%L2       = L2
    p%ar       = ar
    p%az       = az
    p%nBlobs   = nBlobs
    p%blobXc   = blobXc; p%blobYc = blobYc; p%blobZc = blobZc
    p%blobZl   = blobZl; p%blobTwist = blobTwist; p%blobScale = blobScale
    p%width    = width
    p%minor    = minor
    p%major    = major
    p%stretch  = stretch
    p%bGround  = bGround
    p%initDist = initDist//C_NULL_CHAR
    p%initShearA = initShearA
    p%initShearB = initShearB
    p%initShearK = initShearK
    p%pert       = pert
    p%twist       = twist
    p%fRestart = fRestart

    p%nCycle        = nCycle
    p%dt0           = dt0
    p%dtMin         = dtMin
    p%maxError      = maxError
    p%zUpdate       = zUpdate
    p%inertia       = inertia
    p%pressure      = pressure
    p%nu            = nu
    p%beta          = beta
    p%jxbAver       = jxbAver
    p%jxbAverWeight = jxbAverWeight
    p%xPeri         = xPeri
    p%yPeri         = yPeri
    p%zPeri         = zPeri
    p%epsilonProf   = epsilonProf
    p%jMethod       = jMethod//C_NULL_CHAR
    p%pMethod       = pMethod//C_NULL_CHAR

    p%nTs            = nTs
    p%dtDump         = dtDump
    p%dtSave         = dtSave
    p%dumpJ          = dumpJ
    p%dumpDetJac     = dumpDetJac
    p%dumpCellVol    = dumpCellVol
    p%dumpConvexity  = dumpConvexity
    p%dumpWedgeMin   = dumpWedgeMin
    p%redB2          = redB2
    p%redB2f         = redB2f
    p%redBMax        = redBMax
    p%redJMax        = redJMax
    p%redJxB_B2Max   = redJxB_B2Max
    p%redEpsilonStar = redEpsilonStar
    p%redErrB_1ez    = redErrB_1ez
    p%redErrXb_XbAn  = redErrXb_XbAn
    p%redConvex      = redConvex
    p%redWedgeMin    = redWedgeMin
    p%redU2          = redU2
    p%redUMax        = redUMax

!   convert initDist string into integer code. to be used on  the GPU
    if (initDist == 'initShearX') then
        p%initDistCode = 0
    endif
    if (initDist == 'centerShift') then
        p%initDistCode = 1
    endif

end subroutine readNamelist
