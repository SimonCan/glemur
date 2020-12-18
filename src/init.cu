// init.cu
//
// Initial condition routines.
//

#include "global.h"
#include "init.h"
#include "blobsDomes.h"
#include "readExternal.h"


// Set the residual magnetic energy for the corresponding configurations.
void initResiduals(struct Parameters params, struct Reduction *red)
{
    // Residual magnetic energy.
    if (strncmp(params.bInit, "Pontin09 ", 9) == 0)
        red->B2res = 1*params.Lx*params.Ly*params.Lz*params.ampl;
    if (strncmp(params.bInit, "analytic ", 9) == 0)
        red->B2res = 1*params.Lx*params.Ly*params.Lz*params.ampl;
    if (strncmp(params.bInit, "homZ ", 5) == 0)
        red->B2res = 1*params.Lx*params.Ly*params.Lz*params.ampl;
    if (strncmp(params.bInit, "blobs ", 6) == 0)
        red->B2res = 1*params.Lx*params.Ly*params.Lz*params.ampl;
    if (strncmp(params.bInit, "Borromean ", 10) == 0)
        red->B2res = 1*params.Lx*params.Ly*params.Lz*params.bGround;
}


// Create the initial magnetic field B0, the initial grid xb and initial velocity (if needed).
int initState(struct VarsHost h, struct Parameters params, struct Reduction *red)
{
    int  i, j, k, l, b;
    REAL x[params.nx+2], y[params.ny+2], z[params.nz+2];
    REAL tmp, r;

    //
    // magnetic field B0
    //

    if (strncmp(params.bInit, "blobsDome ", 10) == 0)
        initBlobsDome(h, params);
    if (strncmp(params.bInit, "blobsDomeShort ", 15) == 0)
        initBlobsDomeShort(h, params);
    if (strncmp(params.bInit, "blobsDomes2 ", 12) == 0)
        initBlobsDomes2(h, params);
    if (strncmp(params.bInit, "readExternalB0 ", 15) == 0)
        initReadExternalB0(h, params);

    // Include the boundaries.
    for (k = 0; k < params.nz+2; k++) {
        z[k] = params.dz*(k-1) + params.Oz;
        for (j = 0; j < params.ny+2; j++) {
            y[j] = params.dy*(j-1) + params.Oy;
            for (i = 0; i < params.nx+2; i++) {
                x[i] = params.dx*(i-1) + params.Ox;
                // from [1]
                if (strncmp(params.bInit, "Pontin09 ", 9) == 0) {
                    // Field used by Pontin (2009).
                    h.B0[0 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] =
                        -2*params.ampl/(PI*params.ar) * y[j] *
                        (params.phi1*exp(-(x[i]*x[i]+y[j]*y[j])/(params.ar*params.ar)-(z[k]-params.L1)*(z[k]-params.L1)/(params.az*params.az)) +
                         params.phi2*exp(-(x[i]*x[i]+y[j]*y[j])/(params.ar*params.ar)-(z[k]-params.L2)*(z[k]-params.L2)/(params.az*params.az)));
                    h.B0[1 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] =
                        2*params.ampl/(PI*params.ar) * x[i] *
                        (params.phi1*exp(-(x[i]*x[i]+y[j]*y[j])/(params.ar*params.ar)-(z[k]-params.L1)*(z[k]-params.L1)/(params.az*params.az)) +
                         params.phi2*exp(-(x[i]*x[i]+y[j]*y[j])/(params.ar*params.ar)-(z[k]-params.L2)*(z[k]-params.L2)/(params.az*params.az)));
                    h.B0[2 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] = params.ampl;
                }

                // Field for which we know the relaxed state analytically.
                if (strncmp(params.bInit, "analytic ", 9) == 0) {
                    // Field used for testing.
                    h.B0[0 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] =
                        2*params.ampl*z[k]*exp(-(x[i]*x[i]+y[j]*y[j])/(params.ar*params.ar)-z[k]*z[k]/(params.az*params.az))*params.phi1/(params.az*params.az)*y[j];
                    h.B0[1 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] =
                        -2*params.ampl*z[k]*exp(-(x[i]*x[i]+y[j]*y[j])/(params.ar*params.ar)-z[k]*z[k]/(params.az*params.az))*params.phi1/(params.az*params.az)*x[i];
                    h.B0[2 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] = params.ampl;
                }

                // Field for which we know the relaxed state analytically extending in X.
                if (strncmp(params.bInit, "analyticX ", 10) == 0) {
                    // Field used for testing.
                    h.B0[0 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] = params.ampl;
                    h.B0[1 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] =
                        -2*params.ampl*x[i]*exp(-(z[k]*z[k]+y[j]*y[j])/(params.ar*params.ar)-x[i]*x[i]/(params.az*params.az))*params.phi1/(params.az*params.az)*z[k];
                    h.B0[2 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] =
                        2*params.ampl*x[i]*exp(-(z[k]*z[k]+y[j]*y[j])/(params.ar*params.ar)-x[i]*x[i]/(params.az*params.az))*params.phi1/(params.az*params.az)*y[j];
                }

                // Field for which we know the relaxed state analytically extending in Y.
                if (strncmp(params.bInit, "analyticY ", 10) == 0) {
                    // Field used for testing.
                    h.B0[0 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] =
                        2*params.ampl*y[j]*exp(-(x[i]*x[i]+z[k]*z[k])/(params.ar*params.ar)-y[j]*y[j]/(params.az*params.az))*params.phi1/(params.az*params.az)*z[k];
                    h.B0[1 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] = params.ampl;
                    h.B0[2 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] =
                        -2*params.ampl*y[j]*exp(-(x[i]*x[i]+z[k]*z[k])/(params.ar*params.ar)-y[j]*y[j]/(params.az*params.az))*params.phi1/(params.az*params.az)*x[i];
                }

                // Homogeneous magnetic field in z-direction.
                if (strncmp(params.bInit, "homZ ", 5) == 0) {
                    // homogeneous magnetic field
                    h.B0[0 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] = 0.;
                    h.B0[1 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] = 0.;
                    h.B0[2 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] = params.ampl;
                }

                // Blob configuration, like e1, e2 and e3 from Wilmot-Smith ApJ, 696:1339 (2009).
                if (strncmp(params.bInit, "blobs ", 6) == 0) {
                    h.B0[0 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] = 0.;
                    h.B0[1 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] = 0.;
                    h.B0[2 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] = 0.;
                    for (b = 0; b < params.nBlobs; b++) {
                        h.B0[0 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] += 2*params.ampl*params.blobTwist[b]/params.blobScale[b]*
                                exp((-(x[i]-params.blobXc[b])*(x[i]-params.blobXc[b])-(y[j]-params.blobYc[b])*(y[j]-params.blobYc[b]))/(params.blobScale[b]*params.blobScale[b]) -
                                (z[k]-params.blobZc[b])*(z[k]-params.blobZc[b])/(params.blobZl[b]*params.blobZl[b])) *
                                (-(y[j]-params.blobYc[b]));
                        h.B0[1 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] += 2*params.ampl*params.blobTwist[b]/params.blobScale[b]*
                                exp((-(x[i]-params.blobXc[b])*(x[i]-params.blobXc[b])-(y[j]-params.blobYc[b])*(y[j]-params.blobYc[b]))/(params.blobScale[b]*params.blobScale[b]) -
                                (z[k]-params.blobZc[b])*(z[k]-params.blobZc[b])/(params.blobZl[b]*params.blobZl[b])) *
                                (x[i]-params.blobXc[b]);
                    }
                    h.B0[2 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] = params.ampl;
                }

                // Field for the shearing experiments.
                if (strncmp(params.bInit, "sheared ", 8) == 0) {
                    h.B0[0 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] = 0.;
                    h.B0[1 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] = params.ampl*sin(params.initShearK*2*PI*(x[i]-params.Ox)/params.Lx)*0.5;
                    h.B0[2 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] = params.ampl*(1+0.25*pow(cos(params.initShearK*2*PI*(x[i]-params.Ox)/params.Lx),2));
                }

                // Field for null fan configuration from Phys. of Plasm. 12, 072112 (2005).
                if (strncmp(params.bInit, "nullFan ", 8) == 0) {
                    h.B0[0 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] = params.ampl*x[i]/2;
                    h.B0[1 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] = params.ampl*(y[j]/2-params.pert*z[k]);
                    h.B0[2 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] = -params.ampl*z[k];
                }

                // Field for from ApJ 756:7 (6pp), 2012.
                if (strncmp(params.bInit, "fanSeparatrix ", 14) == 0) {
                    tmp = -0.6/sqrt(pow(x[i]*x[i] + pow(y[j]-0.02,2) + pow(z[k]+1.4,2),3));
                    h.B0[0 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] = params.ampl*x[i]*tmp;
                    h.B0[1 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] = params.ampl*(y[j]-0.02)*tmp;
                    h.B0[2 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] = params.ampl*(1+tmp*(z[k]+1.4));
                }

                // Twisted tube.
                if (strncmp(params.bInit, "twisted ", 8) == 0) {
//                    tmp = exp(-(pow(x[i]-params.pert*exp(-pow(z[k],2)/(params.az*params.az)),2)+pow(y[j],2))/(params.ar*params.ar));
                    tmp = exp(-(pow(x[i],2)+pow(y[j],2))/(params.ar*params.ar));
                    h.B0[0 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] = params.ampl*(tmp*y[j]*params.twist - 2*params.pert*z[k]*exp(-z[k]*z[k]/(params.az*params.az))/(params.az*params.az));
                    h.B0[1 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] = -params.ampl*tmp*x[i]*params.twist;
                    h.B0[2 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] = params.ampl;
                }

                // Twisted tubes suggested by G. Hornig 2015.
                if (strncmp(params.bInit, "tubeSetA ", 9) == 0) {
                    r = sqrt(pow(x[i], 2) + pow(y[j], 2));
                    if (r > params.ar) {
                        h.B0[0 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] = 0;
                        h.B0[1 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] = 0;
                        h.B0[2 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] = 0;
                    }
                    else {
                        if (r > 0) {
                            tmp = 4*pow(r/params.ar,3)/params.ar;    // B_p
                            h.B0[0 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] = params.ampl*tmp*y[j]/r;
                            h.B0[1 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] = -params.ampl*tmp*x[i]/r;
                            h.B0[2 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] = params.ampl*(sin(PI*r/(2*params.ar)) + r*PI/(2*params.ar)*cos(PI*r/2/params.ar))/r;
                        }
                        else {
                            h.B0[0 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] = 0;
                            h.B0[1 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] = 0;
                            h.B0[2 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] = params.ampl*PI/params.ar;
                        }
                    }
                }
                if (strncmp(params.bInit, "tubeSetB ", 9) == 0) {
                    r = sqrt(pow(x[i], 2) + pow(y[j], 2));
                    if (r > params.ar) {
                        h.B0[0 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] = 0;
                        h.B0[1 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] = 0;
                        h.B0[2 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] = params.ampl/params.ar;
                    }
                    else {
                        if (r > 0) {
                            tmp = 4*pow(1-pow(r/params.ar,2),2)*r/(params.ar*params.ar);    // B_p
                            h.B0[0 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] = params.ampl*tmp*y[j]/r;
                            h.B0[1 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] = -params.ampl*tmp*x[i]/r;
                            h.B0[2 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] =
                                    params.ampl*(sin(PI/2*pow(r/params.ar,2)) + pow(r/params.ar,2)*PI*cos(PI/2*pow(r/params.ar,2)))/r;
                        }
                        else {
                            h.B0[0 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] = 0;
                            h.B0[1 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] = 0;
                            h.B0[2 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] = 0;
                        }
                    }
                }

                // Hopf field as described in Smiet (2015) 10.1103/PhysRevLett.115.095001.
                if (strncmp(params.bInit, "hopf ", 5) == 0) {
                    tmp = 4*params.ampl*pow(params.ar,4)/(PI*pow((params.ar*params.ar+x[i]*x[i]+y[j]*y[j]+z[k]*z[k]),3));
                    h.B0[0 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] = tmp*2*(params.phi2*params.ar*y[j]-params.phi1*x[i]*z[k]);
                    h.B0[1 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] = -tmp*2*(params.phi2*params.ar*x[i]+params.phi1*y[j]*z[k]);
                    h.B0[2 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] = tmp*params.phi1*(-params.ar*params.ar+x[i]*x[i]+y[j]*y[j]-z[k]*z[k]);
                    h.B0[2 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] += params.bGround;
                }

        // Discontinuous field in z-direction with kink into x.
        if (strncmp(params.bInit, "discont ", 8) == 0) {
            if (z[k] > params.az) {
                h.B0[0 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] = params.pert;
            }
            else {
                h.B0[0 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] = 0;
            }
            h.B0[1 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] = 0;
            h.B0[2 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] = params.ampl;
        }

                // Set the initial grid to undistorted.
                h.xb[0 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] = x[i];
                h.xb[1 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] = y[j];
                h.xb[2 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] = z[k];
            }
        }
    }

    // Borromean rings
    if (strncmp(params.bInit, "Borromean ", 10) == 0) {
        REAL dEllipseParam, dCircleParam, dCircleRadius;
        REAL ellipseParam, circleParam, circleRadius;
        REAL ellipsePos[3], circlePos[3], tangent[3], normal[3];
        REAL len;
        int *nCompute; // array containing the weighting factors for the field smoothing

        // Compute the step lengths such that no grid cell is left out.
        dEllipseParam = min(min(params.dx, params.dy), params.dz) / (params.major+params.width/2.) / params.stretch / 4.;
        dCircleParam = min(min(params.dx, params.dy), params.dz)/(params.width/2.) / params.stretch / 4.;
        dCircleRadius = min(min(params.dx, params.dy), params.dz) / 4.;

        // Initialize the magnetic field to 0.
        memset(h.B0, 0, sizeof(h.B0));

        nCompute = (int *)malloc(params.nx*params.ny*params.nz*sizeof(*(nCompute)));
        if (nCompute == NULL) { printf("error: could not allocate memory for nCompute\n"); return -1; }
        memset(nCompute, 0, sizeof(nCompute));

        for (b = 0; b < 3; b++) {
            ellipseParam = 0.;
            while(ellipseParam <= 2*PI) {
                if (b == 0) {
                    ellipsePos[0] = params.major*sin(ellipseParam);
                    ellipsePos[1] = params.minor*cos(ellipseParam);
                    ellipsePos[2] = 0;
                    tangent[0] = params.major*cos(ellipseParam);
                    tangent[1] = -params.minor*sin(ellipseParam);
                    tangent[2] = 0;
                }
                if (b == 1) {
                    ellipsePos[0] = params.minor*cos(ellipseParam);
                    ellipsePos[1] = 0;
                    ellipsePos[2] = params.major*sin(ellipseParam);
                    tangent[0] = -params.minor*sin(ellipseParam);
                    tangent[1] = 0;
                    tangent[2] = params.major*cos(ellipseParam);
                }
                if (b == 2) {
                    ellipsePos[0] = 0;
                    ellipsePos[1] = params.major*sin(ellipseParam);
                    ellipsePos[2] = params.minor*cos(ellipseParam);
                    tangent[0] = 0;
                    tangent[1] = params.major*cos(ellipseParam);
                    tangent[2] = -params.minor*sin(ellipseParam);
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
                while (circleRadius <= params.width/2.) {
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
                        i = int((circlePos[0]*params.stretch - params.Ox)*params.dx1) + 1;
                        j = int((circlePos[1]*params.stretch - params.Oy)*params.dy1) + 1;
                        k = int((circlePos[2]*params.stretch - params.Oz)*params.dz1) + 1;

                        if ((i >= 0) && (j >= 0) && (k >= 0) && (i < (params.nx+2)) && (j < (params.ny+2)) && (k < (params.nz+2)))
                            for (l = 0; l < 3; l++) {
                                h.B0[l + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] += tangent[l]*params.ampl*
                                        (exp(-(2*circleRadius/params.width)*(2*circleRadius/params.width))-exp(-1.)) / (1-exp(-1.));
                                nCompute[i + j*(params.nx+2) + k*(params.nx+2)*(params.ny+2)] += 1;
                            }

                        circleParam = circleParam + dCircleParam;
                    }
                    circleRadius = circleRadius + dCircleRadius;
                }
                ellipseParam = ellipseParam + dEllipseParam;
            }
        }

        // Add homogeneous magnetic field and do the averaging.
        for (k = 0; k < params.nz+2; k++)
            for (j = 0; j < params.ny+2; j++)
                for (i = 0; i < params.nx+2; i++) {
                    if (nCompute[i + j*(params.nx+2) + k*(params.nx+2)*(params.ny+2)] > 1)
                        for (l = 0; l < 3; l++)
                            h.B0[l + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] /= nCompute[i + j*(params.nx+2) + k*(params.nx+2)*(params.ny+2)];
                    h.B0[2 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] += params.bGround;
                }
    }


    //
    // velocity field uu
    //

    // Exclude the boundaries.
    if (params.inertia == 1) {
        if (strncmp(params.uInit, "nil ", 4) == 0)
            memset(h.B0, 0, sizeof(h.B0));

        // Adapt for own initial velocity field.
//        for (k = 0; k < params.nz; k++) {
//            z[k] = params.dz*k + params.Oz;
//            for (j = 0; j < params.ny; j++) {
//                y[j] = params.dy*j + params.Oy;
//                for (i = 0; i < params.nx; i++) {
//                    x[i] = params.dx*i + params.Ox;
//                    if (strncmp(params.uInit, "nil ", 4) == 0) {
//                        h.uu[0 + i*3 + j*params.nx*3 + k*params.nx*params.ny*3] = 0;
//                        h.uu[1 + i*3 + j*params.nx*3 + k*params.nx*params.ny*3] = 0;
//                        h.uu[2 + i*3 + j*params.nx*3 + k*params.nx*params.ny*3] = 0;
//                    }
//                }
//            }
//        }
    }

    return 0;
}


// Add a distortion to the initial grid xb. Note that B0 refers to the undistorted grid.
int initDistortion(REAL *xb, struct Parameters params)
{
    int  i, j, k;
    REAL x[params.nx+2], y[params.ny+2], z[params.nz+2], yy;

    if (strncmp(params.initDist, "none ", 5) != 0) {
        // Include the boundaries.
        for (k = 0; k < params.nz+2; k++) {
            z[k] = params.dz*(k-1) + params.Oz;
            for (j = 0; j < params.ny+2; j++) {
                y[j] = params.dy*(j-1) + params.Oy;
                for (i = 0; i < params.nx+2; i++) {
                    x[i] = params.dx*(i-1) + params.Ox;

                    // Sinusoidal shear.
                    if (strncmp(params.initDist, "initShearX ", 11) == 0) {
                        xb[1 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] -= params.initShearA * sin(params.initShearK*2*PI*(x[i]+params.Ox-params.dx/2)/(params.Lx+params.dx)) * z[k];
                        yy = xb[1 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3];
                        xb[0 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] -= params.initShearB * sin(params.initShearK*2*PI*(yy+params.Oy-params.dy/2)/(params.Ly+params.dy)) * z[k];
                    }

                    // Sinusoidal shear, works best for boxes of -1 < xyz < 1.
                    if (strncmp(params.initDist, "centerShift ", 12) == 0) {
                        xb[1 + i*3 + j*(params.nx+2)*3 + k*(params.nx+2)*(params.ny+2)*3] +=
                            params.initShearA * exp(-params.initShearK*x[i]*x[i])*(1-y[j]*y[j])*(exp(-1*z[k]+1) - 5.0804508111226507);
                    }
                }
            }
        }
    }
    return 0;
}

