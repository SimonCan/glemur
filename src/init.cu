// init.cu
//
// Initial condition routines.
//

#include "global.h"
#include "init.h"
#include "blobsDomes.h"


// Set the residual magnetic energy for the corresponding configurations.
void initResiduals(struct Parameters p, struct Reduction *red)
{
    // Residual magnetic energy.
    if (strncmp(p.bInit, "Pontin09 ", 9) == 0)
        red->B2res = 1*p.Lx*p.Ly*p.Lz*p.ampl;
    if (strncmp(p.bInit, "analytic ", 9) == 0)
        red->B2res = 1*p.Lx*p.Ly*p.Lz*p.ampl;
    if (strncmp(p.bInit, "homZ ", 5) == 0)
        red->B2res = 1*p.Lx*p.Ly*p.Lz*p.ampl;
    if (strncmp(p.bInit, "blobs ", 6) == 0)
        red->B2res = 1*p.Lx*p.Ly*p.Lz*p.ampl;
    if (strncmp(p.bInit, "Borromean ", 10) == 0)
        red->B2res = 1*p.Lx*p.Ly*p.Lz*p.bGround;
}


// Create the initial magnetic field B0, the initial grid xb and initial velocity (if needed).
int initState(struct VarsHost h, struct Parameters p, struct Reduction *red)
{
    int  i, j, k, l, b;
    REAL x[p.nx+2], y[p.ny+2], z[p.nz+2];
    REAL tmp, r;

    //
    // magnetic field B0
    //

    if (strncmp(p.bInit, "blobsDome ", 10) == 0)
        initBlobsDome(h, p);
    if (strncmp(p.bInit, "blobsDomeShort ", 15) == 0)
        initBlobsDomeShort(h, p);
    if (strncmp(p.bInit, "blobsDomes2 ", 12) == 0)
        initBlobsDomes2(h, p);

    // Include the boundaries.
    for (k = 0; k < p.nz+2; k++) {
        z[k] = p.dz*(k-1) + p.Oz;
        for (j = 0; j < p.ny+2; j++) {
            y[j] = p.dy*(j-1) + p.Oy;
            for (i = 0; i < p.nx+2; i++) {
                x[i] = p.dx*(i-1) + p.Ox;
                // from [1]
                if (strncmp(p.bInit, "Pontin09 ", 9) == 0) {
                    // Field used by Pontin (2009).
                    h.B0[0 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] =
                        -2*p.ampl/(PI*p.ar) * y[j] *
                        (p.phi1*exp(-(x[i]*x[i]+y[j]*y[j])/(p.ar*p.ar)-(z[k]-p.L1)*(z[k]-p.L1)/(p.az*p.az)) +
                         p.phi2*exp(-(x[i]*x[i]+y[j]*y[j])/(p.ar*p.ar)-(z[k]-p.L2)*(z[k]-p.L2)/(p.az*p.az)));
                    h.B0[1 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] =
                        2*p.ampl/(PI*p.ar) * x[i] *
                        (p.phi1*exp(-(x[i]*x[i]+y[j]*y[j])/(p.ar*p.ar)-(z[k]-p.L1)*(z[k]-p.L1)/(p.az*p.az)) +
                         p.phi2*exp(-(x[i]*x[i]+y[j]*y[j])/(p.ar*p.ar)-(z[k]-p.L2)*(z[k]-p.L2)/(p.az*p.az)));
                    h.B0[2 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] = p.ampl;
                }

                // Field for which we know the relaxed state analytically.
                if (strncmp(p.bInit, "analytic ", 9) == 0) {
                    // Field used for testing.
                    h.B0[0 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] =
                        2*p.ampl*z[k]*exp(-(x[i]*x[i]+y[j]*y[j])/(p.ar*p.ar)-z[k]*z[k]/(p.az*p.az))*p.phi1/(p.az*p.az)*y[j];
                    h.B0[1 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] =
                        -2*p.ampl*z[k]*exp(-(x[i]*x[i]+y[j]*y[j])/(p.ar*p.ar)-z[k]*z[k]/(p.az*p.az))*p.phi1/(p.az*p.az)*x[i];
                    h.B0[2 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] = p.ampl;
                }

                // Field for which we know the relaxed state analytically extending in X.
                if (strncmp(p.bInit, "analyticX ", 10) == 0) {
                    // Field used for testing.
                    h.B0[0 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] = p.ampl;
                    h.B0[1 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] =
                        -2*p.ampl*x[i]*exp(-(z[k]*z[k]+y[j]*y[j])/(p.ar*p.ar)-x[i]*x[i]/(p.az*p.az))*p.phi1/(p.az*p.az)*z[k];
                    h.B0[2 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] =
                        2*p.ampl*x[i]*exp(-(z[k]*z[k]+y[j]*y[j])/(p.ar*p.ar)-x[i]*x[i]/(p.az*p.az))*p.phi1/(p.az*p.az)*y[j];
                }

                // Field for which we know the relaxed state analytically extending in Y.
                if (strncmp(p.bInit, "analyticY ", 10) == 0) {
                    // Field used for testing.
                    h.B0[0 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] =
                        2*p.ampl*y[j]*exp(-(x[i]*x[i]+z[k]*z[k])/(p.ar*p.ar)-y[j]*y[j]/(p.az*p.az))*p.phi1/(p.az*p.az)*z[k];
                    h.B0[1 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] = p.ampl;
                    h.B0[2 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] =
                        -2*p.ampl*y[j]*exp(-(x[i]*x[i]+z[k]*z[k])/(p.ar*p.ar)-y[j]*y[j]/(p.az*p.az))*p.phi1/(p.az*p.az)*x[i];
                }

                // Homogeneous magnetic field in z-direction.
                if (strncmp(p.bInit, "homZ ", 5) == 0) {
                    // homogeneous magnetic field
                    h.B0[0 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] = 0.;
                    h.B0[1 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] = 0.;
                    h.B0[2 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] = p.ampl;
                }

                // Blob configuration, like e1, e2 and e3 from Wilmot-Smith ApJ, 696:1339 (2009).
                if (strncmp(p.bInit, "blobs ", 6) == 0) {
                    h.B0[0 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] = 0.;
                    h.B0[1 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] = 0.;
                    h.B0[2 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] = 0.;
                    for (b = 0; b < p.nBlobs; b++) {
                        h.B0[0 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] += 2*p.ampl*p.blobTwist[b]/p.blobScale[b]*
                                exp((-(x[i]-p.blobXc[b])*(x[i]-p.blobXc[b])-(y[j]-p.blobYc[b])*(y[j]-p.blobYc[b]))/(p.blobScale[b]*p.blobScale[b]) -
                                (z[k]-p.blobZc[b])*(z[k]-p.blobZc[b])/(p.blobZl[b]*p.blobZl[b])) *
                                (-(y[j]-p.blobYc[b]));
                        h.B0[1 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] += 2*p.ampl*p.blobTwist[b]/p.blobScale[b]*
                                exp((-(x[i]-p.blobXc[b])*(x[i]-p.blobXc[b])-(y[j]-p.blobYc[b])*(y[j]-p.blobYc[b]))/(p.blobScale[b]*p.blobScale[b]) -
                                (z[k]-p.blobZc[b])*(z[k]-p.blobZc[b])/(p.blobZl[b]*p.blobZl[b])) *
                                (x[i]-p.blobXc[b]);
                    }
                    h.B0[2 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] = p.ampl;
                }

                // Field for the shearing experiments.
                if (strncmp(p.bInit, "sheared ", 8) == 0) {
                    h.B0[0 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] = 0.;
                    h.B0[1 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] = p.ampl*sin(p.initShearK*2*PI*(x[i]-p.Ox)/p.Lx)*0.5;
                    h.B0[2 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] = p.ampl*(1+0.25*pow(cos(p.initShearK*2*PI*(x[i]-p.Ox)/p.Lx),2));
                }

                // Field for null fan configuration from Phys. of Plasm. 12, 072112 (2005).
                if (strncmp(p.bInit, "nullFan ", 8) == 0) {
                    h.B0[0 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] = p.ampl*x[i]/2;
                    h.B0[1 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] = p.ampl*(y[j]/2-p.pert*z[k]);
                    h.B0[2 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] = -p.ampl*z[k];
                }

                // Field for from ApJ 756:7 (6pp), 2012.
                if (strncmp(p.bInit, "fanSeparatrix ", 14) == 0) {
                    tmp = -0.6/sqrt(pow(x[i]*x[i] + pow(y[j]-0.02,2) + pow(z[k]+1.4,2),3));
                    h.B0[0 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] = p.ampl*x[i]*tmp;
                    h.B0[1 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] = p.ampl*(y[j]-0.02)*tmp;
                    h.B0[2 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] = p.ampl*(1+tmp*(z[k]+1.4));
                }

                // Twisted tube.
                if (strncmp(p.bInit, "twisted ", 8) == 0) {
//                    tmp = exp(-(pow(x[i]-p.pert*exp(-pow(z[k],2)/(p.az*p.az)),2)+pow(y[j],2))/(p.ar*p.ar));
                    tmp = exp(-(pow(x[i],2)+pow(y[j],2))/(p.ar*p.ar));
                    h.B0[0 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] = p.ampl*(tmp*y[j]*p.twist - 2*p.pert*z[k]*exp(-z[k]*z[k]/(p.az*p.az))/(p.az*p.az));
                    h.B0[1 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] = -p.ampl*tmp*x[i]*p.twist;
                    h.B0[2 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] = p.ampl;
                }

                // Twisted tubes suggested by G. Hornig 2015.
                if (strncmp(p.bInit, "tubeSetA ", 9) == 0) {
                    r = sqrt(pow(x[i], 2) + pow(y[j], 2));
                    if (r > p.ar) {
                        h.B0[0 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] = 0;
                        h.B0[1 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] = 0;
                        h.B0[2 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] = 0;
                    }
                    else {
                        if (r > 0) {
                            tmp = 4*pow(r/p.ar,3)/p.ar;    // B_p
                            h.B0[0 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] = p.ampl*tmp*y[j]/r;
                            h.B0[1 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] = -p.ampl*tmp*x[i]/r;
                            h.B0[2 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] = p.ampl*(sin(PI*r/(2*p.ar)) + r*PI/(2*p.ar)*cos(PI*r/2/p.ar))/r;
                        }
                        else {
                            h.B0[0 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] = 0;
                            h.B0[1 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] = 0;
                            h.B0[2 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] = p.ampl*PI/p.ar;
                        }
                    }
                }
                if (strncmp(p.bInit, "tubeSetB ", 9) == 0) {
                    r = sqrt(pow(x[i], 2) + pow(y[j], 2));
                    if (r > p.ar) {
                        h.B0[0 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] = 0;
                        h.B0[1 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] = 0;
                        h.B0[2 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] = p.ampl/p.ar;
                    }
                    else {
                        if (r > 0) {
                            tmp = 4*pow(1-pow(r/p.ar,2),2)*r/(p.ar*p.ar);    // B_p
                            h.B0[0 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] = p.ampl*tmp*y[j]/r;
                            h.B0[1 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] = -p.ampl*tmp*x[i]/r;
                            h.B0[2 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] =
                                    p.ampl*(sin(PI/2*pow(r/p.ar,2)) + pow(r/p.ar,2)*PI*cos(PI/2*pow(r/p.ar,2)))/r;
                        }
                        else {
                            h.B0[0 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] = 0;
                            h.B0[1 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] = 0;
                            h.B0[2 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] = 0;
                        }
                    }
                }

                // Hopf field as described in Smiet (2015) 10.1103/PhysRevLett.115.095001.
                if (strncmp(p.bInit, "hopf ", 5) == 0) {
                    tmp = 4*p.ampl*pow(p.ar,4)/(PI*pow((p.ar*p.ar+x[i]*x[i]+y[j]*y[j]+z[k]*z[k]),3));
                    h.B0[0 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] = tmp*2*(p.phi2*p.ar*y[j]-p.phi1*x[i]*z[k]);
                    h.B0[1 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] = -tmp*2*(p.phi2*p.ar*x[i]+p.phi1*y[j]*z[k]);
                    h.B0[2 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] = tmp*p.phi1*(-p.ar*p.ar+x[i]*x[i]+y[j]*y[j]-z[k]*z[k]);
                    h.B0[2 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] += p.bGround;
                }

        // Discontinuous field in z-direction with kink into x.
        if (strncmp(p.bInit, "discont ", 8) == 0) {
            if (z[k] > p.az) {
                h.B0[0 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] = p.pert;
            }
            else {
                h.B0[0 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] = 0;
            }
            h.B0[1 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] = 0;
            h.B0[2 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] = p.ampl;
        }

                // Set the initial grid to undistorted.
                h.xb[0 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] = x[i];
                h.xb[1 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] = y[j];
                h.xb[2 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] = z[k];
            }
        }
    }

    // Borromean rings
    if (strncmp(p.bInit, "Borromean ", 10) == 0) {
        REAL dEllipseParam, dCircleParam, dCircleRadius;
        REAL ellipseParam, circleParam, circleRadius;
        REAL ellipsePos[3], circlePos[3], tangent[3], normal[3];
        REAL len;
        int *nCompute; // array containing the weighting factors for the field smoothing

        // Compute the step lengths such that no grid cell is left out.
        dEllipseParam = min(min(p.dx, p.dy), p.dz) / (p.major+p.width/2.) / p.stretch / 4.;
        dCircleParam = min(min(p.dx, p.dy), p.dz)/(p.width/2.) / p.stretch / 4.;
        dCircleRadius = min(min(p.dx, p.dy), p.dz) / 4.;

        // Initialize the magnetic field to 0.
        memset(h.B0, 0, sizeof(h.B0));

        nCompute = (int *)malloc(p.nx*p.ny*p.nz*sizeof(*(nCompute)));
        if (nCompute == NULL) { printf("error: could not allocate memory for nCompute\n"); return -1; }
        memset(nCompute, 0, sizeof(nCompute));

        for (b = 0; b < 3; b++) {
            ellipseParam = 0.;
            while(ellipseParam <= 2*PI) {
                if (b == 0) {
                    ellipsePos[0] = p.major*sin(ellipseParam);
                    ellipsePos[1] = p.minor*cos(ellipseParam);
                    ellipsePos[2] = 0;
                    tangent[0] = p.major*cos(ellipseParam);
                    tangent[1] = -p.minor*sin(ellipseParam);
                    tangent[2] = 0;
                }
                if (b == 1) {
                    ellipsePos[0] = p.minor*cos(ellipseParam);
                    ellipsePos[1] = 0;
                    ellipsePos[2] = p.major*sin(ellipseParam);
                    tangent[0] = -p.minor*sin(ellipseParam);
                    tangent[1] = 0;
                    tangent[2] = p.major*cos(ellipseParam);
                }
                if (b == 2) {
                    ellipsePos[0] = 0;
                    ellipsePos[1] = p.major*sin(ellipseParam);
                    ellipsePos[2] = p.minor*cos(ellipseParam);
                    tangent[0] = 0;
                    tangent[1] = p.major*cos(ellipseParam);
                    tangent[2] = -p.minor*sin(ellipseParam);
                }
                len = sqrt(tangent[0]*tangent[0]+tangent[1]*tangent[1]+tangent[2]*tangent[2]);
                for (l = 0; l < 3; l++)
                    tangent[l] = tangent[l] / len;

                //  Find vector that is orthonormal to tangent vector.
                if (abs(tangent[0]) <= 0.5) {
                    normal[0] = 0;
                    normal[1] = tangent[2];
                    normal[2] = -tangent[1];
                }
                else if (abs(tangent[1]) <= 0.5) {
                    normal[0] = -tangent[2];
                    normal[1] = 0;
                    normal[2]= tangent[0];
                }
                else {
                    normal[0] = tangent[1];
                    normal[1] = -tangent[0];
                    normal[2] = 0;
                }

                // Normalize the normal vector.
                len = sqrt(normal[0]*normal[0]+normal[1]*normal[1]+normal[2]*normal[2]);
                for (l = 0; l < 3; l++)
                    normal[l] = normal[l] / len;

                circleRadius = 0.;

                // Loop which changes the circle's radius.
                while (circleRadius <= p.width/2.) {
                    circleParam = 0.;

                    // Loop which goes around the circle.
                    while (circleParam <= 2.*PI) {
                        circlePos[0] = ellipsePos[0] + circleRadius *
                                ((tangent[0]*tangent[0]*(1-cos(circleParam))+cos(circleParam))*normal[0] +
                                   (tangent[0]*tangent[1]*(1-cos(circleParam))-tangent[2]*sin(circleParam))*normal[1] +
                                (tangent[0]*tangent[2]*(1-cos(circleParam))+tangent[1]*sin(circleParam))*normal[2]);
                        circlePos[1] = ellipsePos[1] + circleRadius *
                                ((tangent[0]*tangent[1]*(1-cos(circleParam))+tangent[2]*sin(circleParam))*normal[0] +
                                (tangent[1]*tangent[1]*(1-cos(circleParam))+cos(circleParam))*normal[1] +
                                (tangent[1]*tangent[2]*(1-cos(circleParam))-tangent[0]*sin(circleParam))*normal[2]);
                        circlePos[2] = ellipsePos[2] + circleRadius *
                                ((tangent[0]*tangent[2]*(1-cos(circleParam))-tangent[1]*sin(circleParam))*normal[0] +
                                (tangent[1]*tangent[2]*(1-cos(circleParam))+tangent[0]*sin(circleParam))*normal[1] +
                                (tangent[2]*tangent[2]*(1-cos(circleParam))+cos(circleParam))*normal[2]);

                        // Find the corresponding mesh point to this position
                        i = int((circlePos[0]*p.stretch - p.Ox)*p.dx1) + 1;
                        j = int((circlePos[1]*p.stretch - p.Oy)*p.dy1) + 1;
                        k = int((circlePos[2]*p.stretch - p.Oz)*p.dz1) + 1;

                        if ((i >= 0) && (j >= 0) && (k >= 0) && (i < (p.nx+2)) && (j < (p.ny+2)) && (k < (p.nz+2)))
                            for (l = 0; l < 3; l++) {
                                h.B0[l + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] += tangent[l]*p.ampl*
                                        (exp(-(2*circleRadius/p.width)*(2*circleRadius/p.width))-exp(-1.)) / (1-exp(-1.));
                                nCompute[i + j*(p.nx+2) + k*(p.nx+2)*(p.ny+2)] += 1;
                            }

                        circleParam = circleParam + dCircleParam;
                    }
                    circleRadius = circleRadius + dCircleRadius;
                }
                ellipseParam = ellipseParam + dEllipseParam;
            }
        }

        // Add homogeneous magnetic field and do the averaging.
        for (k = 0; k < p.nz+2; k++)
            for (j = 0; j < p.ny+2; j++)
                for (i = 0; i < p.nx+2; i++) {
                    if (nCompute[i + j*(p.nx+2) + k*(p.nx+2)*(p.ny+2)] > 1)
                        for (l = 0; l < 3; l++)
                            h.B0[l + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] /= nCompute[i + j*(p.nx+2) + k*(p.nx+2)*(p.ny+2)];
                    h.B0[2 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] += p.bGround;
                }
    }


    //
    // velocity field uu
    //

    // Exclude the boundaries.
    if (p.inertia == 1) {
        if (strncmp(p.uInit, "nil ", 4) == 0)
            memset(h.B0, 0, sizeof(h.B0));

        // Adapt for own initial velocity field.
//        for (k = 0; k < p.nz; k++) {
//            z[k] = p.dz*k + p.Oz;
//            for (j = 0; j < p.ny; j++) {
//                y[j] = p.dy*j + p.Oy;
//                for (i = 0; i < p.nx; i++) {
//                    x[i] = p.dx*i + p.Ox;
//                    if (strncmp(p.uInit, "nil ", 4) == 0) {
//                        h.uu[0 + i*3 + j*p.nx*3 + k*p.nx*p.ny*3] = 0;
//                        h.uu[1 + i*3 + j*p.nx*3 + k*p.nx*p.ny*3] = 0;
//                        h.uu[2 + i*3 + j*p.nx*3 + k*p.nx*p.ny*3] = 0;
//                    }
//                }
//            }
//        }
    }

    return 0;
}


// Add a distortion to the initial grid xb. Note that B0 refers to the undistorted grid.
int initDistortion(REAL *xb, struct Parameters p)
{
    int  i, j, k;
    REAL x[p.nx+2], y[p.ny+2], z[p.nz+2], yy;

    if (strncmp(p.initDist, "none ", 5) != 0) {
        // Include the boundaries.
        for (k = 0; k < p.nz+2; k++) {
            z[k] = p.dz*(k-1) + p.Oz;
            for (j = 0; j < p.ny+2; j++) {
                y[j] = p.dy*(j-1) + p.Oy;
                for (i = 0; i < p.nx+2; i++) {
                    x[i] = p.dx*(i-1) + p.Ox;

                    // Sinusoidal shear.
                    if (strncmp(p.initDist, "initShearX ", 11) == 0) {
                        xb[1 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] -= p.initShearA * sin(p.initShearK*2*PI*(x[i]+p.Ox-p.dx/2)/(p.Lx+p.dx)) * z[k];
                        yy = xb[1 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3];
                        xb[0 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] -= p.initShearB * sin(p.initShearK*2*PI*(yy+p.Oy-p.dy/2)/(p.Ly+p.dy)) * z[k];
                    }

                    // Sinusoidal shear, works best for boxes of -1 < xyz < 1.
                    if (strncmp(p.initDist, "centerShift ", 12) == 0) {
                        xb[1 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3] +=
                            p.initShearA * exp(-p.initShearK*x[i]*x[i])*(1-y[j]*y[j])*(exp(-1*z[k]+1) - 5.0804508111226507);
                    }
                }
            }
        }
    }
    return 0;
}

