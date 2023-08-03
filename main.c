/*
*/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
//#include <iostream>
//#include <cmath>
//#include <fstream>
//#include <algorithm>
//#include <iomanip>
///// ///////////////////////////////////////////////////GRS_JUPITER///////////////////////////////////////////////////////////////
//using namespace std;

#define pi  3.141592653589793115997963468544185161590576171875000
#define GM  126700000000000000.0

// Planet definition
#define r_e   71492000.0
#define r_p   66854000.0
#define omega 0.000176

// Initial conditions definition
#define theta_0     0.128282 
//  7.35 deg
#define phi_0      -0.392699 
// -22.5 deg
#define a_0         0.244346/2.0 
//14 deg long, 10 deg lat
#define b_0         0.174532/2.0
#define phi_umax  -15.30*pi/180.0
#define theta_vmax 15.04*pi/180.0
#define n_0         2.0

// Define for the data type - to be able to change it fast
#define ARRTYPE double

// Access macro
#define AC(x,j,i,n) x[(j)*(n) + (i)]

// Helper functions
ARRTYPE r_z(ARRTYPE phi){ return r_e*r_e/sqrt(r_p*r_p*tan(phi)*tan(phi)+r_e*r_e); }
ARRTYPE r_m(ARRTYPE phi){ return r_z(phi)/(cos(phi) * (sin(phi)*sin(phi)+r_e*r_e/(r_p*r_p)*cos(phi)*cos(phi))); }

// Structure containing mesh data
typedef struct _mesh {
    ARRTYPE *theta_h, *phi_h;
    ARRTYPE *theta_u, *phi_u;
    ARRTYPE *theta_v, *phi_v;
    ARRTYPE *theta_q, *phi_q;

    ARRTYPE *dx_h, *dy_h;
    ARRTYPE *dx_u, *dy_u;
    ARRTYPE *dx_v, *dy_v;
    ARRTYPE *dx_q, *dy_q;
} mesh;

#define UNPACKMESHDIMS(mesh) \
    ARRTYPE *theta_h = mesh->theta_h;\
    ARRTYPE *theta_u = mesh->theta_u;\
    ARRTYPE *theta_v = mesh->theta_v;\
    ARRTYPE *theta_q = mesh->theta_q;\
    ARRTYPE *phi_h   = mesh->phi_h;\
    ARRTYPE *phi_u   = mesh->phi_u;\
    ARRTYPE *phi_v   = mesh->phi_v;\
    ARRTYPE *phi_q   = mesh->phi_q;

#define UNPACKMESHDXDY(mesh) \
    ARRTYPE *dx_h = mesh->dx_h;\
    ARRTYPE *dx_u = mesh->dx_u;\
    ARRTYPE *dx_v = mesh->dx_v;\
    ARRTYPE *dx_q = mesh->dx_q;\
    ARRTYPE *dy_h = mesh->dy_h;\
    ARRTYPE *dy_u = mesh->dy_u;\
    ARRTYPE *dy_v = mesh->dy_v;\
    ARRTYPE *dy_q = mesh->dy_q;

// Allocate/free data for mesh structure
void alloc_mesh(mesh *m, const int I, const int J) {
    // Malloc
    m->theta_h = (ARRTYPE*)calloc(I*J,sizeof(ARRTYPE));
    m->theta_u = (ARRTYPE*)calloc(I*J,sizeof(ARRTYPE));
    m->theta_v = (ARRTYPE*)calloc(I*J,sizeof(ARRTYPE));
    m->theta_q = (ARRTYPE*)calloc(I*J,sizeof(ARRTYPE));

    m->phi_h   = (ARRTYPE*)calloc(I*J,sizeof(ARRTYPE));
    m->phi_u   = (ARRTYPE*)calloc(I*J,sizeof(ARRTYPE));
    m->phi_v   = (ARRTYPE*)calloc(I*J,sizeof(ARRTYPE));
    m->phi_q   = (ARRTYPE*)calloc(I*J,sizeof(ARRTYPE));

    m->dx_h    = (ARRTYPE*)calloc(I*J,sizeof(ARRTYPE));
    m->dx_u    = (ARRTYPE*)calloc(I*J,sizeof(ARRTYPE));
    m->dx_v    = (ARRTYPE*)calloc(I*J,sizeof(ARRTYPE));
    m->dx_q    = (ARRTYPE*)calloc(I*J,sizeof(ARRTYPE));

    m->dy_h    = (ARRTYPE*)calloc(I*J,sizeof(ARRTYPE));
    m->dy_u    = (ARRTYPE*)calloc(I*J,sizeof(ARRTYPE));
    m->dy_v    = (ARRTYPE*)calloc(I*J,sizeof(ARRTYPE));
    m->dy_q    = (ARRTYPE*)calloc(I*J,sizeof(ARRTYPE));
}

void free_mesh(mesh *m) {
    free(m->theta_h);
    free(m->theta_u);
    free(m->theta_v);
    free(m->theta_q);

    free(m->phi_h);
    free(m->phi_u);
    free(m->phi_v);
    free(m->phi_q);

    free(m->dx_h);
    free(m->dx_u);
    free(m->dx_v);
    free(m->dx_q);

    free(m->dy_h);
    free(m->dy_u);
    free(m->dy_v);
    free(m->dy_q);
}

// Generate the mesh data
void do_mesh(mesh *m,const ARRTYPE L_phi, const ARRTYPE L_theta,const int I, const int J) {
    int i, j;
    // Recover mesh ARRTYPEs
    UNPACKMESHDIMS(m)
    UNPACKMESHDXDY(m)

    const ARRTYPE dphi   = L_phi   / (I - 4.0);
    const ARRTYPE dtheta = L_theta / (J - 4.0);

    #pragma acc parallel loop collapse(2) device_type(nvidia) vector_length(32) \
    present(theta_h,phi_h,theta_u,phi_u,theta_v,phi_v,theta_q,phi_q,dx_h,dy_h,dx_u,dy_u,dx_v,dy_v,dx_q,dy_q) 
    for(i=0; i<I; i++) {
        for(j=0; j<J; j++) {

            AC(theta_h,j,i,I) = -0.523599  + dtheta* (j - 2.0 + 1/2.0); // longitude
            AC(theta_u,j,i,I) = -0.523599  + dtheta* (j - 2.0 + 1.0);
            AC(theta_v,j,i,I) = -0.523599  + dtheta* (j - 2.0 + 1/2.0);
            AC(theta_q,j,i,I) = -0.523599  + dtheta* (j-2.0+1.0);
            
            AC(phi_h,j,i,I)   = -0.6108652 + dphi  * (i - 2.0 + 1/2.0); // latitude
            AC(phi_u,j,i,I)   = -0.6108652 + dphi  * (i - 2.0 + 1/2.0);
            AC(phi_v,j,i,I)   = -0.6108652 + dphi  * (i - 2.0 + 1.0);
            AC(phi_q,j,i,I)   = -0.6108652 + dphi  * (i-2.0+1.0);

            AC(dx_h,j,i,I)    = dtheta * r_z(AC(phi_h,j,i,I));
            AC(dy_h,j,i,I)    = dphi   * r_m(AC(phi_h,j,i,I));

            AC(dx_u,j,i,I)    = dtheta * r_z(AC(phi_u,j,i,I));
            AC(dy_u,j,i,I)    = dphi   * r_m(AC(phi_u,j,i,I));
            
            AC(dx_v,j,i,I)    = dtheta * r_z(AC(phi_v,j,i,I));
            AC(dy_v,j,i,I)    = dphi   * r_m(AC(phi_v,j,i,I));
            
            AC(dx_q,j,i,I)    = dtheta * r_z(AC(phi_q,j,i,I));
            AC(dy_q,j,i,I)    = dphi   * r_m(AC(phi_q,j,i,I));
        }
    }
}


typedef struct _flow {
    ARRTYPE *u, *v;
    ARRTYPE *U;   // Velocity + wind speed
    
    ARRTYPE *eta; // perturbation above D
    ARRTYPE *h;   // total height of fluid
    
    ARRTYPE *wind; // Wind speed

    ARRTYPE *q;   // potential vorticity
    ARRTYPE *w;   // relative vorticity

    // Auxiliar arrays for time integration
    ARRTYPE *u_old, *u_new;
    ARRTYPE *v_old, *v_new;
}flow;

#define UNPACKFLOW(flow) \
    ARRTYPE *u     = flow->u;\
    ARRTYPE *v     = flow->v;\
    ARRTYPE *U     = flow->U;\
    ARRTYPE *eta   = flow->eta;\
    ARRTYPE *h     = flow->h;\
    ARRTYPE *q     = flow->q;\
    ARRTYPE *w     = flow->w;\
    ARRTYPE *wind  = flow->wind;\
    ARRTYPE *u_old = flow->u_old;\
    ARRTYPE *u_new = flow->u_new;\
    ARRTYPE *v_old = flow->v_old;\
    ARRTYPE *v_new = flow->v_new;

void alloc_flow(flow *f,const int I, const int J) {
    f->u     = (ARRTYPE*)calloc(I*J,sizeof(ARRTYPE));
    f->v     = (ARRTYPE*)calloc(I*J,sizeof(ARRTYPE));
    f->U     = (ARRTYPE*)calloc(I*J,sizeof(ARRTYPE));
    
    f->eta   = (ARRTYPE*)calloc(I*J,sizeof(ARRTYPE));
    f->h     = (ARRTYPE*)calloc(I*J,sizeof(ARRTYPE));
    
    f->q     = (ARRTYPE*)calloc(I*J,sizeof(ARRTYPE));
    f->w     = (ARRTYPE*)calloc(I*J,sizeof(ARRTYPE));

    f->wind  = (ARRTYPE*)calloc(I,sizeof(ARRTYPE));

    f->u_old = (ARRTYPE*)calloc(I*J,sizeof(ARRTYPE));
    f->u_new = (ARRTYPE*)calloc(I*J,sizeof(ARRTYPE));
    f->v_old = (ARRTYPE*)calloc(I*J,sizeof(ARRTYPE));
    f->v_new = (ARRTYPE*)calloc(I*J,sizeof(ARRTYPE));
}

void free_flow(flow *f) {
    free(f->u);
    free(f->v);
    free(f->U);
    free(f->eta);
    free(f->h);
    free(f->q);
    free(f->w);
    free(f->wind);
    free(f->u_old);
    free(f->u_new);
    free(f->v_old);
    free(f->v_new);
}


ARRTYPE psi(ARRTYPE theta, ARRTYPE phi) { return ((theta-theta_0)*(theta-theta_0)/(a_0*a_0)+(phi-phi_0)*(phi-phi_0)/(b_0*b_0)); }
ARRTYPE phi_c(ARRTYPE phi) { return atan(r_p*r_p/(r_e*r_e)*tan(phi)); } //planetocentric lat
ARRTYPE radi(ARRTYPE phi){ return r_e*r_p/sqrt(r_e*r_e*sin(phi_c(phi))*sin(phi_c(phi))+r_p*r_p*cos(phi_c(phi))*cos(phi_c(phi))); }
ARRTYPE g_function(ARRTYPE phi){ return GM/(radi(phi_c(phi))*radi(phi_c(phi))) - omega*omega*radi(phi_c(phi))*cos(phi_c(phi)); }
ARRTYPE f_function(ARRTYPE phi) { return 2.0*omega*sin(phi); }

void initialConditions(flow *f, mesh *m, const ARRTYPE D, const int I, const int J) {
    int i, j;
    // Recover array types
    UNPACKFLOW(f)
    UNPACKMESHDIMS(m)

    const ARRTYPE A = r_m(phi_umax)*abs(f_function(phi_umax))*b_0*126.5/(2.0*n_0*g_function(phi_umax))*pow(1.0-1.0/(2.0*n_0),(1.0-2.0*n_0)/(2.0*n_0))*exp(1.0-1.0/(2.0*n_0));


    #pragma acc parallel loop collapse(2) device_type(nvidia) vector_length(32) \
    present(theta_h,phi_h,theta_u,phi_u,theta_v,phi_v,eta,u,v,u_old,v_old,h) 
    for(i=2; i<I-2; i++) {
        for(j=2; j<J-2; j++) {

            AC(eta,j,i,I) = A*exp(-pow(psi(AC(theta_h,j,i,I),AC(theta_h,j,i,I)),n_0));
            AC(u,j,i,I)   = 2.0*g_function(AC(phi_u,j,i,I))*n_0/(f_function(AC(phi_u,j,i,I))*
                            r_m(AC(phi_u,j,i,I))*b_0*b_0)*(AC(phi_u,j,i,I)-phi_0)*
                            pow(psi(AC(theta_u,j,i,I),AC(phi_u,j,i,I)),n_0-1.0)*
                            (A*exp(-pow(psi(AC(theta_u,j,i,I),AC(phi_u,j,i,I)),n_0)));

            AC(v,j,i,I)   = 2.0*g_function(AC(phi_v,j,i,I))*n_0/(f_function(AC(phi_v,j,i,I))*
                            r_z(AC(phi_v,j,i,I))*a_0*a_0)*(AC(theta_v,j,i,I)-theta_0)*
                            pow(psi(AC(theta_v,j,i,I),AC(phi_v,j,i,I)),n_0-1.0)*
                            (A*exp(-pow(psi(AC(theta_v,j,i,I),AC(phi_v,j,i,I)),n_0)));

            AC(u_old,j,i,I) = AC(u,j,i,I);
            AC(v_old,j,i,I) = AC(v,j,i,I);

            AC(h,j,i,I) = D + AC(eta,j,i,I);
        }
    }
}

// CANAL Boundary conditions DPC for left and right FSC for top and bottom
void set_BC_momentum(flow *f, const int I, const int J) {
    int i, j;
    UNPACKFLOW(f)

    // Top and bottom
    #pragma acc parallel loop device_type(nvidia) vector_length(32) \
    present(u,v,u_old,v_old) 
    for(j=0; j<J; j++){
        //top
        AC(u,j,0,I)       = AC(u,j,2,I);
        AC(u,j,1,I)       = AC(u,j,2,I);
        AC(v,j,0,I)       = 0.0;
        AC(v,j,1,I)       = 0.0;
        AC(u_old,j,0,I)   = AC(u_old,j,2,I);
        AC(u_old,j,1,I)   = AC(u_old,j,2,I);
        AC(v_old,j,0,I)   = 0.0;
        AC(v_old,j,1,I)   = 0.0;

        //bottom
        AC(u,j,I-1,I)     = AC(u,j,I-3,I);
        AC(u,j,I-2,I)     = AC(u,j,I-3,I);
        AC(v,j,I-1,I)     = 0.0;
        AC(v,j,I-2,I)     = 0.0;
        AC(v,j,I-3,I)     = 0.0;
        AC(u_old,j,I-1,I) = AC(u_old,j,I-3,I);
        AC(u_old,j,I-2,I) = AC(u_old,j,I-3,I);
        AC(v_old,j,I-1,I) = 0.0;
        AC(v_old,j,I-2,I) = 0.0;
        AC(v_old,j,I-3,I) = 0.0;
    }

    // Left and right
    #pragma acc parallel loop device_type(nvidia) vector_length(32) \
    present(eta,u,v,u_old,v_old,h) 
    for(i=0; i<I; i++){
        // left
        AC(u,0,i,I)       = AC(u,J-4,i,I);
        AC(u,1,i,I)       = AC(u,J-3,i,I);
        AC(v,0,i,I)       = AC(v,J-4,i,I);
        AC(v,1,i,I)       = AC(v,J-3,i,I);
        AC(u_old,0,i,I)   = AC(u_old,J-4,i,I);
        AC(u_old,1,i,I)   = AC(u_old,J-3,i,I);
        AC(v_old,0,i,I)   = AC(v_old,J-4,i,I);
        AC(v_old,1,i,I)   = AC(v_old,J-3,i,I);

        // right
        AC(u,J-1,i,I)     = AC(u,3,i,I);
        AC(u,J-2,i,I)     = AC(u,2,i,I);
        AC(v,J-1,i,I)     = AC(v,3,i,I);
        AC(v,J-2,i,I)     = AC(v,2,i,I);
        AC(u_old,J-1,i,I) = AC(u_old,3,i,I);
        AC(u_old,J-2,i,I) = AC(u_old,2,i,I);
        AC(v_old,J-1,i,I) = AC(v_old,3,i,I);
        AC(v_old,J-2,i,I) = AC(v_old,2,i,I);
    }
}

void set_BC_continuity(flow *f, const int I, const int J) {
    int i, j;
    UNPACKFLOW(f)

    // Top and bottom
    #pragma acc parallel loop device_type(nvidia) vector_length(32) \
    present(eta,h) 
    for(j=0; j<J; j++){
        //top
        AC(h,j,0,I)       = 0.0;
        AC(h,j,1,I)       = 0.0;
        AC(eta,j,0,I)     = 0.0;
        AC(eta,j,1,I)     = 0.0;

        //bottom
        AC(h,j,I-1,I)     = 0.0;
        AC(h,j,I-2,I)     = 0.0;
        AC(eta,j,I-1,I)   = 0.0;
        AC(eta,j,I-2,I)   = 0.0;
    }

    // Left and right
    #pragma acc parallel loop device_type(nvidia) vector_length(32) \
    present(eta,h) 
    for(i=0; i<I; i++){
        // left
        AC(h,0,i,I)       = AC(h,J-4,i,I);
        AC(h,1,i,I)       = AC(h,J-3,i,I);
        AC(eta,0,i,I)     = AC(eta,J-4,i,I);
        AC(eta,1,i,I)     = AC(eta,J-3,i,I);

        // right
        AC(h,J-1,i,I)     = AC(h,3,i,I);
        AC(h,J-2,i,I)     = AC(h,2,i,I);
        AC(eta,J-1,i,I)   = AC(eta,3,i,I);
        AC(eta,J-2,i,I)   = AC(eta,2,i,I);
    }
}

void add_winds(flow *f, const int I, const int J) {
    int i, j;
    UNPACKFLOW(f)

    #pragma acc parallel loop collapse(2) device_type(nvidia) vector_length(32) \
    present(U,u,wind) 
    for(j=0; j<J; j++){
        for(i=0; i<I; i++){
            AC(U,j,i,I) = AC(u,j,i,I) + wind[i];
        }
    }
}


// Numerical method - space discretization

ARRTYPE flux_limiter(ARRTYPE r){ return fmax(0.,fmax(fmin(2.*r,1.),fmin(r,2.))); }

void rhs_momentum(ARRTYPE *f_u, ARRTYPE *f_v, flow *f, mesh *m, const ARRTYPE dt, const int I, const int J) {
    int i, j;
    UNPACKMESHDIMS(m)
    UNPACKMESHDXDY(m)
    UNPACKFLOW(f)

    ARRTYPE r_w_u, r_ww_u, r_e_u, r_ee_u, r_s_u, r_ss_u, r_n_u, r_nn_u;
    ARRTYPE r_w_v, r_ww_v, r_e_v, r_ee_v, r_s_v, r_ss_v, r_n_v, r_nn_v;
    ARRTYPE u_w, u_ww, v_w, v_ww;
    ARRTYPE u_e, u_ee, v_e, v_ee;
    ARRTYPE u_s, u_ss, v_s, v_ss;
    ARRTYPE u_n, u_nn, v_n, v_nn;
    ARRTYPE C_w_u, C_ww_u, C_e_u, C_ee_u, C_s_u, C_ss_u, C_n_u, C_nn_u; //C_w positiu, C_ww negatiu
    ARRTYPE C_w_v, C_ww_v, C_e_v, C_ee_v, C_s_v, C_ss_v, C_n_v, C_nn_v;
    ARRTYPE P1u, P2u, Ppu;
    ARRTYPE P1v, P2v, Ppv;

    #pragma acc parallel loop collapse(2) device_type(nvidia) vector_length(32) \
    present(f_u,f_v,U,v,eta,phi_u,phi_v,dx_u,dy_u,dx_v,dy_v)
    for(j=2; j<J-2; j++){
        for(i=2; i<I-2; i++){
            // computation of r parameter for u
            if( AC(U,j,i,I) - AC(U,j-1,i,I) == 0.0 ){
                r_w_u  = 0.0;
                r_ww_u = 0.0;
            }
            else{
                r_w_u  = (AC(U,j-1,i,I) - AC(U,j-2,i,I)) / (AC(U,j,i,I) - AC(U,j-1,i,I));
                r_ww_u = (AC(U,j+1,i,I) - AC(U,j,i,I))   / (AC(U,j,i,I) - AC(U,j-1,i,I));
            }

            if( AC(U,j+1,i,I) - AC(U,j,i,I) == 0.0 ){
                r_e_u  = 0.0;
                r_ee_u = 0.0;
            }
            else{
                r_e_u  = (AC(U,j,i,I)   - AC(U,j-1,i,I)) / (AC(U,j+1,i,I) - AC(U,j,i,I));
                r_ee_u = (AC(U,j+2,i,I) - AC(U,j+1,i,I)) / (AC(U,j+1,i,I) - AC(U,j,i,I));
            }

            if( AC(U,j,i+1,I) - AC(U,j,i,I) == 0.0 ){
                r_s_u  = 0.0;
                r_ss_u = 0.0;
            }
            else{
                r_s_u  = (AC(U,j,i,I)   - AC(U,j,i-1,I)) / (AC(U,j,i+1,I) - AC(U,j,i,I));
                r_ss_u = (AC(U,j,i+2,I) - AC(U,j,i+1,I)) / (AC(U,j,i+1,I) - AC(U,j,i,I));
            }

            if( AC(U,j,i,I) - AC(U,j,i-1,I) == 0.0 ){
                r_n_u  = 0.0;
                r_nn_u = 0.0;
            }
            else{
                r_n_u  = (AC(U,j,i-1,I) - AC(U,j,i-2,I)) / (AC(U,j,i,I) - AC(U,j,i-1,I));
                r_nn_u = (AC(U,j,i+1,I) - AC(U,j,i,I))   / (AC(U,j,i,I) - AC(U,j,i-1,I));
            }

            //computation of r parameter for v
            if( AC(v,j,i,I) - AC(v,j-1,i,I) == 0.0 ){
                r_w_v  = 0.0;
                r_ww_v = 0.0;
            }
            else{
                r_w_v  = (AC(v,j-1,i,I) - AC(v,j-2,i,I)) / (AC(v,j,i,I) - AC(v,j-1,i,I));
                r_ww_v = (AC(v,j+1,i,I) - AC(v,j,i,I))   / (AC(v,j,i,I) - AC(v,j-1,i,I));
            }

            if( AC(v,j+1,i,I) - AC(v,j,i,I) == 0.0 ){
                r_e_v  = 0.0;
                r_ee_v = 0.0;
            }
            else{
                r_e_v  = (AC(v,j,i,I)   - AC(v,j-1,i,I)) / (AC(v,j+1,i,I) - AC(v,j,i,I));
                r_ee_v = (AC(v,j+2,i,I) - AC(v,j+1,i,I)) / (AC(v,j+1,i,I) - AC(v,j,i,I));
            }

            if( AC(v,j,i+1,I) - AC(v,j,i,I) == 0.0 ){
                r_s_v  = 0.0;
                r_ss_v = 0.0;
            }
            else{
                r_s_v  = (AC(v,j,i,I)   - AC(v,j,i-1,I)) / (AC(v,j,i+1,I) - AC(v,j,i,I));
                r_ss_v = (AC(v,j,i+2,I) - AC(v,j,i+1,I)) / (AC(v,j,i+1,I) - AC(v,j,i,I));
            }

            if( AC(v,j,i,I) - AC(v,j,i-1,I) == 0.0 ){
                r_n_v  = 0.0;
                r_nn_v = 0.0;
            }
            else{
                r_n_v  = (AC(v,j,i-1,I) - AC(v,j,i-2,I)) / (AC(v,j,i,I) - AC(v,j,i-1,I));
                r_nn_v = (AC(v,j,i+1,I) - AC(v,j,i,I))   / (AC(v,j,i,I) - AC(v,j,i-1,I));
            }

            //Courant numbers for u and v
            C_w_u  = 1/2.0 * (1/2.0 * (AC(U,j-1,i,I) + fabs(AC(U,j-1,i,I))) + 1/2.0 * (AC(U,j,i,I)     + fabs(AC(U,j,i,I))))     * dt / AC(dx_u,j,i,I);
            C_ww_u = 1/2.0 * (1/2.0 * (AC(U,j-1,i,I) - fabs(AC(U,j-1,i,I))) + 1/2.0 * (AC(U,j,i,I)     - fabs(AC(U,j,i,I))))     * dt / AC(dx_u,j,i,I);
            C_e_u  = 1/2.0 * (1/2.0 * (AC(U,j,i,I)   + fabs(AC(U,j,i,I)))   + 1/2.0 * (AC(U,j+1,i,I)   + fabs(AC(U,j+1,i,I))))   * dt / AC(dx_u,j,i,I);
            C_ee_u = 1/2.0 * (1/2.0 * (AC(U,j,i,I)   - fabs(AC(U,j,i,I)))   + 1/2.0 * (AC(U,j+1,i,I)   - fabs(AC(U,j+1,i,I))))   * dt / AC(dx_u,j,i,I);
            C_s_u  = 1/2.0 * (1/2.0 * (AC(v,j,i,I)   + fabs(AC(v,j,i,I)))   + 1/2.0 * (AC(v,j+1,i,I)   + fabs(AC(v,j+1,i,I))))   * dt / AC(dy_u,j,i,I);
            C_ss_u = 1/2.0 * (1/2.0 * (AC(v,j,i,I)   - fabs(AC(v,j,i,I)))   + 1/2.0 * (AC(v,j+1,i,I)   - fabs(AC(v,j+1,i,I))))   * dt / AC(dy_u,j,i,I);
            C_n_u  = 1/2.0 * (1/2.0 * (AC(v,j,i-1,I) + fabs(AC(v,j,i-1,I))) + 1/2.0 * (AC(v,j+1,i-1,I) + fabs(AC(v,j+1,i-1,I)))) * dt / AC(dy_u,j,i,I);
            C_nn_u = 1/2.0 * (1/2.0 * (AC(v,j,i-1,I) - fabs(AC(v,j,i-1,I))) + 1/2.0 * (AC(v,j+1,i-1,I) - fabs(AC(v,j+1,i-1,I)))) * dt / AC(dy_u,j,i,I);

            C_w_v  = 1/2.0 * (1/2.0 * (AC(U,j-1,i+1,I) + fabs(AC(U,j-1,i+1,I))) + 1/2.0 * (AC(U,j-1,i,I) + fabs(AC(U,j-1,i,I)))) * dt / AC(dx_v,j,i,I);
            C_ww_v = 1/2.0 * (1/2.0 * (AC(U,j-1,i+1,I) - fabs(AC(U,j-1,i+1,I))) + 1/2.0 * (AC(U,j-1,i,I) - fabs(AC(U,j-1,i,I)))) * dt / AC(dx_v,j,i,I);
            C_e_v  = 1/2.0 * (1/2.0 * (AC(U,j,i+1,I)   + fabs(AC(U,j,i+1,I)))   + 1/2.0 * (AC(U,j,i,I)   + fabs(AC(U,j,i,I))))   * dt / AC(dx_v,j,i,I);
            C_ee_v = 1/2.0 * (1/2.0 * (AC(U,j,i+1,I)   - fabs(AC(U,j,i+1,I)))   + 1/2.0 * (AC(U,j,i,I)   - fabs(AC(U,j,i,I))))   * dt / AC(dx_v,j,i,I);
            C_s_v  = 1/2.0 * (1/2.0 * (AC(v,j,i+1,I)   + fabs(AC(v,j,i+1,I)))   + 1/2.0 * (AC(v,j,i,I)   + fabs(AC(v,j,i,I))))   * dt / AC(dy_v,j,i,I);
            C_ss_v = 1/2.0 * (1/2.0 * (AC(v,j,i+1,I)   - fabs(AC(v,j,i+1,I)))   + 1/2.0 * (AC(v,j,i,I)   - fabs(AC(v,j,i,I))))   * dt / AC(dy_v,j,i,I);
            C_n_v  = 1/2.0 * (1/2.0 * (AC(v,j,i,I)     + fabs(AC(v,j,i,I)))     + 1/2.0 * (AC(v,j,i-1,I) + fabs(AC(v,j,i-1,I)))) * dt / AC(dy_v,j,i,I);
            C_nn_v = 1/2.0 * (1/2.0 * (AC(v,j,i,I)     - fabs(AC(v,j,i,I)))     + 1/2.0 * (AC(v,j,i-1,I) - fabs(AC(v,j,i-1,I)))) * dt / AC(dy_v,j,i,I);


            u_w  = AC(U,j-1,i,I) + 1/2.0 * flux_limiter(r_w_u)  * (1.0 - C_w_u)  * (AC(U,j,i,I)   - AC(U,j-1,i,I));
            u_ww = AC(U,j,i,I)   - 1/2.0 * flux_limiter(r_ww_u) * (1.0 + C_ww_u) * (AC(U,j,i,I)   - AC(U,j-1,i,I));
            u_e  = AC(U,j,i,I)   + 1/2.0 * flux_limiter(r_e_u)  * (1.0 - C_e_u)  * (AC(U,j+1,i,I) - AC(U,j,i,I));
            u_ee = AC(U,j+1,i,I) - 1/2.0 * flux_limiter(r_ee_u) * (1.0 + C_ee_u) * (AC(U,j+1,i,I) - AC(U,j,i,I));
            u_s  = AC(U,j,i,I)   + 1/2.0 * flux_limiter(r_s_u)  * (1.0 - C_s_u)  * (AC(U,j,i+1,I) - AC(U,j,i,I));
            u_ss = AC(U,j,i+1,I) - 1/2.0 * flux_limiter(r_ss_u) * (1.0 + C_ss_u) * (AC(U,j,i+1,I) - AC(U,j,i,I));
            u_n  = AC(U,j,i-1,I) + 1/2.0 * flux_limiter(r_n_u)  * (1.0 - C_n_u)  * (AC(U,j,i,I)   - AC(U,j,i-1,I));
            u_nn = AC(U,j,i,I)   - 1/2.0 * flux_limiter(r_nn_u) * (1.0 + C_nn_u) * (AC(U,j,i,I)   - AC(U,j,i-1,I));

            v_w  = AC(v,j-1,i,I) + 1/2.0 * flux_limiter(r_w_v)  * (1.0 - C_w_v)  * (AC(v,j,i,I)   - AC(v,j-1,i,I));
            v_ww = AC(v,j,i,I)   - 1/2.0 * flux_limiter(r_ww_v) * (1.0 + C_ww_v) * (AC(v,j,i,I)   - AC(v,j-1,i,I));
            v_e  = AC(v,j,i,I)   + 1/2.0 * flux_limiter(r_e_v)  * (1.0 - C_e_v)  * (AC(v,j+1,i,I) - AC(v,j,i,I));
            v_ee = AC(v,j+1,i,I) - 1/2.0 * flux_limiter(r_ee_v) * (1.0 + C_ee_v) * (AC(v,j+1,i,I) - AC(v,j,i,I));
            v_s  = AC(v,j,i,I)   + 1/2.0 * flux_limiter(r_s_v)  * (1.0 - C_s_v)  * (AC(v,j,i+1,I) - AC(v,j,i,I));
            v_ss = AC(v,j,i+1,I) - 1/2.0 * flux_limiter(r_ss_v) * (1.0 + C_ss_v) * (AC(v,j,i+1,I) - AC(v,j,i,I));
            v_n  = AC(v,j,i-1,I) + 1/2.0 * flux_limiter(r_n_v)  * (1.0 - C_n_v)  * (AC(v,j,i,I)   - AC(v,j,i-1,I));
            v_nn = AC(v,j,i,I)   - 1/2.0 * flux_limiter(r_nn_v) * (1.0 + C_nn_v) * (AC(v,j,i,I)   - AC(v,j,i-1,I));

            
            P1u = C_w_u * u_w + C_ww_u * u_ww - C_e_u * u_e - C_ee_u * u_ee - C_s_u * u_s - C_ss_u * u_ss + C_n_u * u_n + C_nn_u * u_nn;
            P2u = AC(U,j,i,I)*((AC(U,j+1,i,I)-AC(U,j-1,i,I))/(2.0*AC(dx_u,j,i,I)) + ((AC(v,j,i,I)+AC(v,j+1,i,I))-(AC(v,j,i-1,I)+AC(v,j+1,i-1,I)))/(2.0*AC(dy_u,j,i,I)));
            Ppu = -g_function(AC(phi_u,j,i,I))*((AC(eta,j+1,i,I)-AC(eta,j,i,I))/AC(dx_u,j,i,I));

            AC(f_u,j,i,I) = P1u/dt + P2u + Ppu + (sin(AC(phi_u,j,i,I))/r_z(AC(phi_u,j,i,I)) * AC(u,j,i,I) * (AC(v,j,i,I)+AC(v,j+1,i,I)+AC(v,j+1,i-1,I)+AC(v,j,i-1,I))/4.0);

            
            P1v = C_w_v * v_w + C_ww_v * v_ww - C_e_v * v_e - C_ee_v * v_ee - C_s_v * v_s - C_ss_v * v_ss + C_n_v * v_n + C_nn_v * v_nn;
            P2v = AC(v,j,i,I)*((AC(v,j,i+1,I)-AC(v,j,i-1,I))/(2.0*AC(dy_v,j,i,I)) + ((AC(U,j,i,I)+AC(U,j,i+1,I))-(AC(U,j-1,i+1,I)+AC(U,j-1,i,I)))/(2.0*AC(dx_v,j,i,I)));
            Ppv = -g_function(AC(phi_v,j,i,I))*((AC(eta,j,i+1,I)-AC(eta,j,i,I))/AC(dy_v,j,i,I));

            AC(f_v,j,i,I) = P1v/dt + P2v + Ppv - (sin(AC(phi_v,j,i,I))/r_z(AC(phi_v,j,i,I)) * (AC(u,j,i,I)+AC(u,j-1,i,I)+AC(u,j-1,i+1,I)+AC(u,j,i+1,I))/4.0*(AC(u,j,i,I)+AC(u,j-1,i,I)+AC(u,j-1,i+1,I)+AC(u,j,i+1,I))/4.0);
        }
    }
}



// Numerical method - time discretization

void advance_Euler(flow *f, ARRTYPE *f_u, ARRTYPE *f_v, ARRTYPE dt, const int I, const int J) {
    int i, j;
    UNPACKFLOW(f)

    #pragma acc parallel loop collapse(2) device_type(nvidia) vector_length(32) \
    present(u,v,f_u,f_v)     
    for(j=2; j<J-2; j++){
        for(i=2; i<I-2; i++){
            AC(u,j,i,I) += dt * AC(f_u,j,i,I);
            AC(v,j,i,I) += dt * AC(f_v,j,i,I);
        }
    }    
}




//float flux_limiter(float r){
//    float llista2[]={2.0*r,1.0};
//    float llista3[]={r,2.0};
//    float llista[] = {0.0, *min_element(llista2,llista2+2), *min_element(llista3,llista3+2)};
//    float flux_lim = *max_element(llista,llista+3);
//    return flux_lim;
//}
//



/*
float closest(float vec[1459], float value){
    int var;
    for(int x=0; x<1459; x++){
        if(vec[x]>=value){
            var = x;
        }
        else{
            break;
        }
    }
    return var;
}
*/


int main(){

    // parameters
    const int J = 204, I = 104; //J horitzontal, I vertical
    const ARRTYPE L_phi   = 0.610865;  // 35 deg
    const ARRTYPE L_theta = 1.221730;  // 70 deg
    const ARRTYPE D       = 1000.0;
    const ARRTYPE dt      = 30.0;
    
    int i; int j; int n;

    // Generate mesh
    mesh m; alloc_mesh(&m,I,J); // Allocate memory
    do_mesh(&m,L_phi,L_theta,I,J);

    // Generate flow
    flow f; alloc_flow(&f,I,J);

    // Set up initial conditions
    initialConditions(&f,&m,D,I,J);
    set_BC_momentum(&f,I,J);
    set_BC_continuity(&f,I,J);

    // Start time loop
    for(n=0; n<201601; n++){
        // Add wind contribution
        add_winds(&f,I,J);
    

    }

////variables
//float r_w_u; float r_ww_u;
//float r_e_u; float r_ee_u;
//float r_s_u; float r_ss_u;
//float r_n_u; float r_nn_u;
//
//float r_w_v; float r_ww_v;
//float r_e_v; float r_ee_v;
//float r_s_v; float r_ss_v;
//float r_n_v; float r_nn_v;
//
//float r_w_h; float r_ww_h;
//float r_e_h; float r_ee_h;
//float r_s_h; float r_ss_h;
//float r_n_h; float r_nn_h;
//
//
//float u_w; float u_ww;
//float v_w; float v_ww;
//float h_w; float h_ww;
//
//float u_e; float u_ee;
//float v_e; float v_ee;
//float h_e; float h_ee;
//
//float u_s; float u_ss;
//float v_s; float v_ss;
//float h_s; float h_ss;
//
//float u_n; float u_nn;
//float v_n; float v_nn;
//float h_n; float h_nn;
//
//
//float C_w_u; float C_ww_u; float C_e_u; float C_ee_u; float C_s_u; float C_ss_u; float C_n_u; float C_nn_u; //C_w positiu, C_ww negatiu
//float C_w_v; float C_ww_v; float C_e_v; float C_ee_v; float C_s_v; float C_ss_v; float C_n_v; float C_nn_v;
//float C_w_h; float C_ww_h; float C_e_h; float C_ee_h; float C_s_h; float C_ss_h; float C_n_h; float C_nn_h;
//
//
//float u[J][I];
//float v[J][I];
//
//float U[J][I];
//
//float u_old[J][I];
//float u_new[J][I];
//float v_old[J][I];
//float v_new[J][I];
//
//float eta[J][I]; //perturbation above D
//float h[J][I]; //total height of fluid
//
//double q[J][I]; //potential vorticity
//double w[J][I]; // relative vorticity
//
//
//float P1u[J][I]; float P1v[J][I];
//float P2u[J][I]; float P2v[J][I];
//float Ppu[J][I]; float Ppv[J][I];
//float Ph[J][I];
//
//// functions to be integrated
//float f_u[J][I]; float f_u_old[J][I]; float f_u_2old[J][I];
//float f_v[J][I]; float f_v_old[J][I]; float f_v_2old[J][I];
//float f_h[J][I]; float f_h_old[J][I]; float f_h_2old[J][I];
//
//
//// ellipsoidal coordinates and differentials
//float theta_h[J][I]; float phi_h[J][I];
//float theta_u[J][I]; float phi_u[J][I];
//float theta_v[J][I]; float phi_v[J][I];
//float theta_q[J][I]; float phi_q[J][I];
//
//
//const float dphi   = L_phi/(I-4.0);
//const float dtheta = L_theta/(J-4.0);
//
//float dx_h[J][I]; float dy_h[J][I];
//float dx_u[J][I]; float dy_u[J][I];
//float dx_v[J][I]; float dy_v[J][I];
//float dx_q[J][I]; float dy_q[J][I];
//
//
//#pragma acc parallel loop collapse(2) device_type(nvidia) vector_length(32) \
//present(theta_h,phi_h,theta_u,phi_u,theta_v,phi_v,theta_q,phi_q,dx_h,dy_h,dx_u,dy_u,dx_v,dy_v,dx_q,dy_q) 
//for(i=0; i<I; i++){
//    for(j=0; j<J; j++){
//
//        theta_h[j][i] = -0.523599  + dtheta*(j - 2.0 + 1/2.0); // longitude
//        phi_h  [j][i] = -0.6108652 + dphi  *(i - 2.0 + 1/2.0); // latitude
//
//        theta_u[j][i] = -0.523599  + dtheta*(j - 2.0 + 1.0);
//        phi_u  [j][i] = -0.6108652 + dphi  *(i - 2.0 + 1/2.0);
//
//        theta_v[j][i] = -0.523599  + dtheta*(j - 2.0 + 1/2.0);
//        phi_v  [j][i] = -0.6108652 + dphi  *(i - 2.0 + 1.0);
//
//        theta_q[j][i] = -0.523599  + dtheta*(j-2.0+1.0);
//        phi_q  [j][i] = -0.6108652 + dphi  *(i-2.0+1.0);
//
//        dx_h[j][i] = dtheta * r_z(phi_h[j][i]);
//        dy_h[j][i] = dphi * r_m(phi_h[j][i]);
//
//        dx_u[j][i] = dtheta * r_z(phi_u[j][i]);
//        dy_u[j][i] = dphi * r_m(phi_u[j][i]);
//
//        dx_v[j][i] = dtheta * r_z(phi_v[j][i]);
//        dy_v[j][i] = dphi * r_m(phi_v[j][i]);
//
//        dx_q[j][i] = dtheta * r_z(phi_q[j][i]);
//        dy_q[j][i] = dphi * r_m(phi_q[j][i]);
//    }
//}
//
///*
//// Introduction of zonal winds files and ARRTYPEs
//ifstream latitude_file;
//latitude_file.open("latitude_file");
//if (!latitude_file.is_open()){
//    cout << "error \n";
//    return 0;
//}
//
//
//ifstream zonal_winds_file;
//zonal_winds_file.open("zonal_winds_file");
//if (!zonal_winds_file.is_open()){
//    cout << "error \n";
//    return 0;
//}
//
//
//int Z = 1459; int z = 0;
//float z_winds_file[Z];
//float lat_file[Z];
//float z_winds[I];
//
//
//for( z=0; z<Z; z++){
//    zonal_winds_file >> z_winds_file[z];
//    latitude_file >> lat_file[z];
//    lat_file[z] = lat_file[z]*3.14159265359/180.0;
//}
//
//zonal_winds_file.close();
//latitude_file.close();
//
//
//int val;
//for( i=2; i<I-2; i++){         //interpolacio lineal dels vents zonals
//    val=closest(lat_file, phi_u[2][i]);
//    z_winds[i] = (z_winds_file[val]+z_winds_file[val+1])/2.0;
//}
//
//*/
//
////Initial conditions
//const long double pi = 3.14159265359;
//const float theta_0 = 0.128282; //7.35 deg
//const float phi_0 = -0.392699; //-22.5 deg
//const float a_0 = 0.244346/2.0; //14 deg long, 10 deg lat
//const float b_0 =  0.174532/2.0;
//const float phi_umax = -15.30*pi/180.0;
//const float theta_vmax = 15.04*pi/180.0;
//const float n_0 = 2.0;
//const float A = r_m(phi_umax)*abs(f(phi_umax))*b_0*126.5/(2.0*n_0*g_function(phi_umax))*pow(1.0-1.0/(2.0*n_0),(1.0-2.0*n_0)/(2.0*n_0))*exp(1.0-1.0/(2.0*n_0));
//
//
//for(i=2; i<I-2; i++){
//    for(j=2; j<J-2; j++){
//
//        eta[j][i] = A*exp(-pow(psi(theta_h[j][i],phi_h[j][i]),n_0));
//
//        u[j][i] = 2.0*g_function(phi_u[j][i])*n_0/(f(phi_u[j][i])*r_m(phi_u[j][i])*b_0*b_0)*
//        (phi_u[j][i]-phi_0)*pow(psi(theta_u[j][i],phi_u[j][i]),n_0-1.0)*
//        (A*exp(-pow(psi(theta_u[j][i],phi_u[j][i]),n_0)));
//
//        v[j][i] =  2.0*g_function(phi_v[j][i])*n_0/(f(phi_v[j][i])*r_z(phi_v[j][i])*a_0*a_0)*
//        (theta_v[j][i]-theta_0)*pow(psi(theta_v[j][i],phi_v[j][i]),n_0-1.0)*
//        (A*exp(-pow(psi(theta_v[j][i],phi_v[j][i]),n_0)));
//
//        u_old[j][i] = u[j][i];
//        v_old[j][i] = v[j][i];
//
//        h[j][i] = D + eta[j][i];
//    }
//}
//
//
//FILE*output;
//char buffer[30];
//
//
//
////CANAL Boundary conditions DPC for left and right FSC for top and bottom
//
//for(j=0; j<J; j++){
//    //top
//    h[j][0] = 0.0;
//    h[j][1] = 0.0;
//
//    eta[j][0] = 0.0;
//    eta[j][1] = 0.0;
//
//    u[j][0] = u[j][2];
//    u[j][1] = u[j][2];
//
//    //z_winds[0] = z_winds[2];
//    //z_winds[1] = z_winds[2];
//
//    v[j][0] = 0.0;
//    v[j][1] = 0.0;
//
//    u_old[j][0] = u_old[j][2];
//    u_old[j][1] = u_old[j][2];
//
//    v_old[j][0] = 0.0;
//    v_old[j][1] = 0.0;
//
//    //bottom
//    h[j][I-1] = 0.0;
//    h[j][I-2] = 0.0;
//
//    eta[j][I-1] = 0.0;  //Recordar que: AC(eta,j,I-1,I) == eta[j][I-1] 
//    eta[j][I-2] = 0.0;
//
//    u[j][I-1] = u[j][I-3];
//    u[j][I-2] = u[j][I-3];
//
//    //z_winds[I-1] = z_winds[I-3];
//    //z_winds[I-2] = z_winds[I-3];
//
//    v[j][I-1] = 0.0;
//    v[j][I-2] = 0.0;
//    v[j][I-3] = 0.0;
//
//    u_old[j][I-1] = u_old[j][I-3];
//    u_old[j][I-2] = u_old[j][I-3];
//
//    v_old[j][I-1] = 0.0;
//    v_old[j][I-2] = 0.0;
//    v_old[j][I-3] = 0.0;
//}
//
//for(i=0; i<I; i++){
//    //left
//    h[0][i] = h[J-4][i];
//    h[1][i] = h[J-3][i];
//
//    eta[0][i] = eta[J-4][i];
//    eta[1][i] = eta[J-3][i];
//
//    u[0][i] = u[J-4][i];
//    u[1][i] = u[J-3][i];
//
//    v[0][i] = v[J-4][i];
//    v[1][i] = v[J-3][i];
//
//    u_old[0][i] = u_old[J-4][i];
//    u_old[1][i] = u_old[J-3][i];
//
//    v_old[0][i] = v_old[J-4][i];
//    v_old[1][i] = v_old[J-3][i];
//
//    //right
//    h[J-1][i] = h[3][i];
//    h[J-2][i] = h[2][i];
//
//    eta[J-1][i] = eta[3][i];
//    eta[J-2][i] = eta[2][i];
//
//    u[J-1][i] = u[3][i];
//    u[J-2][i] = u[2][i];
//
//    v[J-1][i] = v[3][i];
//    v[J-2][i] = v[2][i];
//
//    u_old[J-1][i] = u_old[3][i];
//    u_old[J-2][i] = u_old[2][i];
//
//    v_old[J-1][i] = v_old[3][i];
//    v_old[J-2][i] = v_old[2][i];
//}
//
//
//
//for(n=0; n<201601; n++){
//
//    for(j=0; j<J; j++){
//        for(i=0; i<I; i++){
//            U[j][i] = u[j][i]; // + z_winds[i];
//        }
//    }
//
//    for(j=2; j<J-2; j++){
//        for(i=2; i<I-2; i++){
//
//            //computation of r parameter for u
//
//
//            if( U[j][i] - U[j-1][i] == 0.0 ){
//                r_w_u = 0.0;
//                r_ww_u = 0.0;
//            }
//            else{
//                r_w_u = (U[j-1][i] - U[j-2][i]) / (U[j][i] - U[j-1][i]);
//                r_ww_u = (U[j+1][i] - U[j][i]) / (U[j][i] - U[j-1][i]);
//            }
//
//            if( U[j+1][i] - U[j][i] == 0.0 ){
//                r_e_u = 0.0;
//                r_ee_u = 0.0;
//            }
//            else{
//                r_e_u = (U[j][i] - U[j-1][i]) / (U[j+1][i] - U[j][i]);
//                r_ee_u = (U[j+2][i] - U[j+1][i]) / (U[j+1][i] - U[j][i]);
//            }
//
//            if( U[j][i+1] - U[j][i] == 0.0 ){
//                r_s_u = 0.0;
//                r_ss_u = 0.0;
//            }
//            else{
//                r_s_u = (U[j][i] - U[j][i-1]) / (U[j][i+1] - U[j][i]);
//                r_ss_u = (U[j][i+2] - U[j][i+1]) / (U[j][i+1] - U[j][i]);
//            }
//
//            if( U[j][i] - U[j][i-1] == 0.0 ){
//                r_n_u = 0.0;
//                r_nn_u = 0.0;
//            }
//            else{
//                r_n_u = (U[j][i-1] - U[j][i-2]) / (U[j][i] - U[j][i-1]);
//                r_nn_u = (U[j][i+1] - U[j][i]) / (U[j][i] - U[j][i-1]);
//            }
//
//            //computation of r parameter for v
//
//            if( v[j][i] - v[j-1][i] == 0.0 ){
//                r_w_v = 0.0;
//                r_ww_v = 0.0;
//            }
//            else{
//                r_w_v = (v[j-1][i] - v[j-2][i]) / (v[j][i] - v[j-1][i]);
//                r_ww_v = (v[j+1][i] - v[j][i]) / (v[j][i] - v[j-1][i]);
//            }
//
//            if( v[j+1][i] - v[j][i] == 0.0 ){
//                r_e_v = 0.0;
//                r_ee_v = 0.0;
//            }
//            else{
//                r_e_v = (v[j][i] - v[j-1][i]) / (v[j+1][i] - v[j][i]);
//                r_ee_v = (v[j+2][i] - v[j+1][i]) / (v[j+1][i] - v[j][i]);
//            }
//
//            if( v[j][i+1] - v[j][i] == 0.0 ){
//                r_s_v = 0.0;
//                r_ss_v = 0.0;
//            }
//            else{
//                r_s_v = (v[j][i] - v[j][i-1]) / (v[j][i+1] - v[j][i]);
//                r_ss_v = (v[j][i+2] - v[j][i+1]) / (v[j][i+1] - v[j][i]);
//            }
//
//            if( v[j][i] - v[j][i-1] == 0.0 ){
//                r_n_v = 0.0;
//                r_nn_v = 0.0;
//            }
//            else{
//                r_n_v = (v[j][i-1] - v[j][i-2]) / (v[j][i] - v[j][i-1]);
//                r_nn_v = (v[j][i+1] - v[j][i]) / (v[j][i] - v[j][i-1]);
//            }
//
//            //Courant numbers for u and v
//
//            C_w_u = 1/2.0 * (1/2.0 * (U[j-1][i] + abs(U[j-1][i])) + 1/2.0 * (U[j][i] + abs(U[j][i]))) * dt / dx_u[j][i];
//            C_ww_u = 1/2.0 * (1/2.0 * (U[j-1][i] - abs(U[j-1][i])) + 1/2.0 * (U[j][i] - abs(U[j][i]))) * dt / dx_u[j][i];
//            C_e_u = 1/2.0 * (1/2.0 * (U[j][i] + abs(U[j][i])) + 1/2.0 * (U[j+1][i] + abs(U[j+1][i]))) * dt / dx_u[j][i];
//            C_ee_u = 1/2.0 * (1/2.0 * (U[j][i] - abs(U[j][i])) + 1/2.0 * (U[j+1][i] - abs(U[j+1][i]))) * dt / dx_u[j][i];
//            C_s_u = 1/2.0 * (1/2.0 * (v[j][i] + abs(v[j][i])) + 1/2.0 * (v[j+1][i] + abs(v[j+1][i]))) * dt / dy_u[j][i];
//            C_ss_u = 1/2.0 * (1/2.0 * (v[j][i] - abs(v[j][i])) + 1/2.0 * (v[j+1][i] - abs(v[j+1][i]))) * dt / dy_u[j][i];
//            C_n_u = 1/2.0 * (1/2.0 * (v[j][i-1] + abs(v[j][i-1])) + 1/2.0 * (v[j+1][i-1] + abs(v[j+1][i-1]))) * dt / dy_u[j][i];
//            C_nn_u = 1/2.0 * (1/2.0 * (v[j][i-1] - abs(v[j][i-1])) + 1/2.0 * (v[j+1][i-1] - abs(v[j+1][i-1]))) * dt / dy_u[j][i];
//
//            C_w_v = 1/2.0 * (1/2.0 * (U[j-1][i+1] + abs(U[j-1][i+1])) + 1/2.0 * (U[j-1][i] + abs(U[j-1][i]))) * dt / dx_v[j][i];
//            C_ww_v = 1/2.0 * (1/2.0 * (U[j-1][i+1] - abs(U[j-1][i+1])) + 1/2.0 * (U[j-1][i] - abs(U[j-1][i]))) * dt / dx_v[j][i];
//            C_e_v = 1/2.0 * (1/2.0 * (U[j][i+1] + abs(U[j][i+1])) + 1/2.0 * (U[j][i] + abs(U[j][i]))) * dt / dx_v[j][i];
//            C_ee_v = 1/2.0 * (1/2.0 * (U[j][i+1] - abs(U[j][i+1])) + 1/2.0 * (U[j][i] - abs(U[j][i]))) * dt / dx_v[j][i];
//            C_s_v = 1/2.0 * (1/2.0 * (v[j][i+1] + abs(v[j][i+1])) + 1/2.0 * (v[j][i] + abs(v[j][i]))) * dt / dy_v[j][i];
//            C_ss_v = 1/2.0 * (1/2.0 * (v[j][i+1] - abs(v[j][i+1])) + 1/2.0 * (v[j][i] - abs(v[j][i]))) * dt / dy_v[j][i];
//            C_n_v = 1/2.0 * (1/2.0 * (v[j][i] + abs(v[j][i])) + 1/2.0 * (v[j][i-1] + abs(v[j][i-1]))) * dt / dy_v[j][i];
//            C_nn_v = 1/2.0 * (1/2.0 * (v[j][i] - abs(v[j][i])) + 1/2.0 * (v[j][i-1] - abs(v[j][i-1]))) * dt / dy_v[j][i];
//
//
//            u_w = U[j-1][i] + 1/2.0 * flux_limiter(r_w_u) * (1.0 - C_w_u) * (U[j][i] - U[j-1][i]);
//            u_ww = U[j][i] - 1/2.0 * flux_limiter(r_ww_u) * (1.0 + C_ww_u) * (U[j][i] - U[j-1][i]);
//            u_e = U[j][i] + 1/2.0 * flux_limiter(r_e_u) * (1.0 - C_e_u) * (U[j+1][i] - U[j][i]);
//            u_ee = U[j+1][i] - 1/2.0 * flux_limiter(r_ee_u) * (1.0 + C_ee_u) * (U[j+1][i] - U[j][i]);
//            u_s = U[j][i] + 1/2.0 * flux_limiter(r_s_u) * (1.0 - C_s_u) * (U[j][i+1] - U[j][i]);
//            u_ss = U[j][i+1] - 1/2.0 * flux_limiter(r_ss_u) * (1.0 + C_ss_u) * (U[j][i+1] - U[j][i]);
//            u_n = U[j][i-1] + 1/2.0 * flux_limiter(r_n_u) * (1.0 - C_n_u) * (U[j][i] - U[j][i-1]);
//            u_nn = U[j][i] - 1/2.0 * flux_limiter(r_nn_u) * (1.0 + C_nn_u) * (U[j][i] - U[j][i-1]);
//
//            v_w = v[j-1][i] + 1/2.0 * flux_limiter(r_w_v) * (1.0 - C_w_v) * (v[j][i] - v[j-1][i]);
//            v_ww = v[j][i] - 1/2.0 * flux_limiter(r_ww_v) * (1.0 + C_ww_v) * (v[j][i] - v[j-1][i]);
//            v_e = v[j][i] + 1/2.0 * flux_limiter(r_e_v) * (1.0 - C_e_v) * (v[j+1][i] - v[j][i]);
//            v_ee = v[j+1][i] - 1/2.0 * flux_limiter(r_ee_v) * (1.0 + C_ee_v) * (v[j+1][i] - v[j][i]);
//            v_s = v[j][i] + 1/2.0 * flux_limiter(r_s_v) * (1.0 - C_s_v) * (v[j][i+1] - v[j][i]);
//            v_ss = v[j][i+1] - 1/2.0 * flux_limiter(r_ss_v) * (1.0 + C_ss_v) * (v[j][i+1] - v[j][i]);
//            v_n = v[j][i-1] + 1/2.0 * flux_limiter(r_n_v) * (1.0 - C_n_v) * (v[j][i] - v[j][i-1]);
//            v_nn = v[j][i] - 1/2.0 * flux_limiter(r_nn_v) * (1.0 + C_nn_v) * (v[j][i] - v[j][i-1]);
//
//
//            P1u[j][i] = C_w_u * u_w + C_ww_u * u_ww - C_e_u * u_e - C_ee_u * u_ee - C_s_u * u_s - C_ss_u * u_ss + C_n_u * u_n + C_nn_u * u_nn;
//            P2u[j][i] = U[j][i]*((U[j+1][i]-U[j-1][i])/(2.0*dx_u[j][i]) + ((v[j][i]+v[j+1][i])-(v[j][i-1]+v[j+1][i-1]))/(2.0*dy_u[j][i]));
//            Ppu[j][i] = -g_function(phi_u[j][i])*((eta[j+1][i]-eta[j][i])/dx_u[j][i]);
//
//            f_u[j][i] = P1u[j][i]/dt + P2u[j][i] + Ppu[j][i] +
//            (sin(phi_u[j][i])/r_z(phi_u[j][i]) * u[j][i] * (v[j][i]+v[j+1][i]+v[j+1][i-1]+v[j][i-1])/4.0);
//
//
//            P1v[j][i] = C_w_v * v_w + C_ww_v * v_ww - C_e_v * v_e - C_ee_v * v_ee - C_s_v * v_s - C_ss_v * v_ss + C_n_v * v_n + C_nn_v * v_nn;
//            P2v[j][i] = v[j][i]*((v[j][i+1]-v[j][i-1])/(2.0*dy_v[j][i]) + ((U[j][i]+U[j][i+1])-(U[j-1][i+1]+U[j-1][i]))/(2.0*dx_v[j][i]));
//            Ppv[j][i] = -g_function(phi_v[j][i])*((eta[j][i+1]-eta[j][i])/dy_v[j][i]);
//
//            f_v[j][i] = P1v[j][i]/dt + P2v[j][i] + Ppv[j][i] -
//            (sin(phi_v[j][i])/r_z(phi_v[j][i]) * (u[j][i]+u[j-1][i]+u[j-1][i+1]+u[j][i+1])/4.0*(u[j][i]+u[j-1][i]+u[j-1][i+1]+u[j][i+1])/4.0);
//        }
//    }
//
//
//    //Momentum eqs time integration
//    switch(n){
//
//        case 0:     //Euler
//
//            for(j=2; j<J-2; j++){
//                for(i=2; i<I-2; i++){
//
//                    u[j][i] += dt * f_u[j][i];
//                    v[j][i] += dt * f_v[j][i];
//
//                    f_u_2old[j][i] = f_u[j][i];
//                    f_v_2old[j][i] = f_v[j][i];
//                }
//            }
//            break;
//
//        case 1:     //Euler
//
//            for(j=2; j<J-2; j++){
//                for(i=2; i<I-2; i++){
//
//                    u[j][i] += dt * f_u[j][i];
//                    v[j][i] += dt * f_v[j][i];
//
//                    f_u_old[j][i] = f_u[j][i];
//                    f_v_old[j][i] = f_v[j][i];
//                }
//            }
//            break;
//
//        default:        //Adam-Bashforth
//
//            for(j=2; j<J-2; j++){
//                for(i=2; i<I-2; i++){
//
//                    u[j][i] += dt * (23.0/12.0 * f_u[j][i] - 4.0/3.0 * f_u_old[j][i] + 5.0/12.0 * f_u_2old[j][i]);
//                    v[j][i] += dt * (23.0/12.0 * f_v[j][i] - 4.0/3.0 * f_v_old[j][i] + 5.0/12.0 * f_v_2old[j][i]);
//
//                    f_u_2old[j][i] = f_u_old[j][i];
//                    f_u_old[j][i] = f_u[j][i];
//
//                    f_v_2old[j][i] = f_v_old[j][i];
//                    f_v_old[j][i] = f_v[j][i];
//                }
//            }
//
//    }
//
//
//    //Canal Boundary conditions
//    for(j=0; j<J; j++){
//        //top
//        u[j][0] = u[j][2];
//        u[j][1] = u[j][2];
//
//        v[j][0] = 0.0;
//        v[j][1] = 0.0;
//
//        u_old[j][0] = u_old[j][2];
//        u_old[j][1] = u_old[j][2];
//
//        v_old[j][0] = 0.0;
//        v_old[j][1] = 0.0;
//
//        //bottom
//        u[j][I-1] = u[j][I-3];
//        u[j][I-2] = u[j][I-3];
//
//        v[j][I-1] = 0.0;
//        v[j][I-2] = 0.0;
//        v[j][I-3] = 0.0;
//
//        u_old[j][I-1] = u_old[j][I-3];
//        u_old[j][I-2] = u_old[j][I-3];
//
//        v_old[j][I-1] = 0.0;
//        v_old[j][I-2] = 0.0;
//        v_old[j][I-3] = 0.0;
//    }
//
//    for(i=0; i<I; i++){
//        //left
//        u[0][i] = u[J-4][i];
//        u[1][i] = u[J-3][i];
//
//        v[0][i] = v[J-4][i];
//        v[1][i] = v[J-3][i];
//
//        u_old[0][i] = u_old[J-4][i];
//        u_old[1][i] = u_old[J-3][i];
//
//        v_old[0][i] = v_old[J-4][i];
//        v_old[1][i] = v_old[J-3][i];
//
//        //right
//        u[J-1][i] = u[3][i];
//        u[J-2][i] = u[2][i];
//
//        v[J-1][i] = v[3][i];
//        v[J-2][i] = v[2][i];
//
//        u_old[J-1][i] = u_old[3][i];
//        u_old[J-2][i] = u_old[2][i];
//
//        v_old[J-1][i] = v_old[3][i];
//        v_old[J-2][i] = v_old[2][i];
//    }
//
//
//    //Coriolis correction
//    for(j=2; j<J-2; j++){
//        for(i=2; i<I-2; i++){
//
//            u_new[j][i] = (u[j][i] - (0.5*f(phi_u[j][i])*dt)*(0.5*f(phi_u[j][i])*dt)*u_old[j][i] + 0.25*f(phi_u[j][i])*dt *
//            (v_old[j][i]+v_old[j+1][i]+v_old[j+1][i-1]+v_old[j][i-1]))/ (1.0 + (0.5*f(phi_u[j][i])*dt)*(0.5*f(phi_u[j][i])*dt));
//
//            v_new[j][i] = (v[j][i] - (0.5*f(phi_v[j][i])*dt)*(0.5*f(phi_v[j][i])*dt)*v_old[j][i] - 0.25*f(phi_v[j][i])*dt *
//            (u_old[j][i]+u_old[j-1][i]+u_old[j-1][i+1]+u_old[j][i+1]))/ (1.0 + (0.5*f(phi_v[j][i])*dt)*(0.5*f(phi_v[j][i])*dt));
//
//            u_old[j][i] = u_new[j][i];
//            v_old[j][i] = v_new[j][i];
//
//            u[j][i] = u_new[j][i];
//            v[j][i] = v_new[j][i];
//        }
//    }
//
//
//    //Canal Boundary conditions
//    for(j=0; j<J; j++){
//        //top
//        u[j][0] = u[j][2];
//        u[j][1] = u[j][2];
//
//        v[j][0] = 0.0;
//        v[j][1] = 0.0;
//
//        u_old[j][0] = u_old[j][2];
//        u_old[j][1] = u_old[j][2];
//
//        v_old[j][0] = 0.0;
//        v_old[j][1] = 0.0;
//
//        //bottom
//        u[j][I-1] = u[j][I-3];
//        u[j][I-2] = u[j][I-3];
//
//        v[j][I-1] = 0.0;
//        v[j][I-2] = 0.0;
//        v[j][I-3] = 0.0;
//
//        u_old[j][I-1] = u_old[j][I-3];
//        u_old[j][I-2] = u_old[j][I-3];
//
//        v_old[j][I-1] = 0.0;
//        v_old[j][I-2] = 0.0;
//        v_old[j][I-3] = 0.0;
//    }
//
//    for(i=0; i<I; i++){
//        //left
//        u[0][i] = u[J-4][i];
//        u[1][i] = u[J-3][i];
//
//        v[0][i] = v[J-4][i];
//        v[1][i] = v[J-3][i];
//
//        u_old[0][i] = u_old[J-4][i];
//        u_old[1][i] = u_old[J-3][i];
//
//        v_old[0][i] = v_old[J-4][i];
//        v_old[1][i] = v_old[J-3][i];
//
//        //right
//        u[J-1][i] = u[3][i];
//        u[J-2][i] = u[2][i];
//
//        v[J-1][i] = v[3][i];
//        v[J-2][i] = v[2][i];
//
//        u_old[J-1][i] = u_old[3][i];
//        u_old[J-2][i] = u_old[2][i];
//
//        v_old[J-1][i] = v_old[3][i];
//        v_old[J-2][i] = v_old[2][i];
//    }
//
//
//
//    for(j=0; j<J; j++){
//        for(i=0; i<I; i++){
//            U[j][i] = u[j][i]; // + z_winds[i];
//        }
//    }
//
//
//    for(j=2; j<J-2; j++){
//        for(i=2; i<I-2; i++){
//
//            //r parameter for h
//
//            if( h[j][i] - h[j-1][i] == 0.0 ){
//                r_w_h = 0.0;
//                r_ww_h = 0.0;
//            }
//            else{
//                r_w_h = (h[j-1][i] - h[j-2][i]) / (h[j][i] - h[j-1][i]);
//                r_ww_h= (h[j+1][i] - h[j][i]) / (h[j][i] - h[j-1][i]);
//            }
//
//            if( h[j+1][i] - h[j][i] == 0.0 ){
//                r_e_h = 0.0;
//                r_ee_h = 0.0;
//            }
//            else{
//                r_e_h = (h[j][i] - h[j-1][i]) / (h[j+1][i] - h[j][i]);
//                r_ee_h = (h[j+2][i] - h[j+1][i]) / (h[j+1][i] - h[j][i]);
//            }
//
//            if( h[j][i+1] - h[j][i] == 0.0 ){
//                r_s_h = 0.0;
//                r_ss_h = 0.0;
//            }
//            else{
//                r_s_h = (h[j][i] - h[j][i-1]) / (h[j][i+1] - h[j][i]);
//                r_ss_h = (h[j][i+2] - h[j][i+1]) / (h[j][i+1] - h[j][i]);
//            }
//
//            if( h[j][i] - h[j][i-1] == 0.0 ){
//                r_n_h = 0.0;
//                r_nn_h = 0.0;
//            }
//            else{
//                r_n_h = (h[j][i-1] - h[j][i-2]) / (h[j][i] - h[j][i-1]);
//                r_nn_h = (h[j][i+1] - h[j][i]) / (h[j][i] - h[j][i-1]);
//            }
//
//            //Courant numbers for h
//
//            C_w_h = 1/2.0 *  (U[j-1][i] + abs(U[j-1][i])) * dt / dx_h[j][i];
//            C_ww_h = 1/2.0 * (U[j-1][i] - abs(U[j-1][i])) * dt / dx_h[j][i];
//            C_e_h = 1/2.0 * (U[j][i] + abs(U[j][i])) * dt / dx_h[j][i];
//            C_ee_h = 1/2.0 *  (U[j][i] - abs(U[j][i])) * dt / dx_h[j][i];
//            C_s_h = 1/2.0 *  (v[j][i] + abs(v[j][i])) * dt / dy_h[j][i];
//            C_ss_h = 1/2.0 * (v[j][i] - abs(v[j][i])) * dt / dy_h[j][i];
//            C_n_h = 1/2.0 * (v[j][i-1] + abs(v[j][i-1])) * dt / dy_h[j][i];
//            C_nn_h = 1/2.0 * (v[j][i-1] - abs(v[j][i-1])) * dt / dy_h[j][i];
//
//
//            h_w = h[j-1][i] + 0.5 * flux_limiter(r_w_h) * (1.0 - C_w_h) * (h[j][i] - h[j-1][i]);
//            h_ww = h[j][i] - 0.5 * flux_limiter(r_ww_h) * (1.0 + C_ww_h) * (h[j][i] - h[j-1][i]);
//            h_e = h[j][i] + 0.5 * flux_limiter(r_e_h) * (1.0 - C_e_h) * (h[j+1][i] - h[j][i]);
//            h_ee = h[j+1][i] - 0.5 * flux_limiter(r_ee_h) * (1.0 + C_ee_h) * (h[j+1][i] - h[j][i]);
//            h_s = h[j][i] + 0.5 * flux_limiter(r_s_h) * (1.0 - C_s_h) * (h[j][i+1] - h[j][i]);
//            h_ss = h[j][i+1] - 0.5 * flux_limiter(r_ss_h) * (1.0 + C_ss_h) * (h[j][i+1] - h[j][i]);
//            h_n = h[j][i-1] + 0.5 * flux_limiter(r_n_h) * (1.0 - C_n_h) * (h[j][i] - h[j][i-1]);
//            h_nn = h[j][i] - 0.5 * flux_limiter(r_nn_h) * (1.0 + C_nn_h) * (h[j][i] - h[j][i-1]);
//
//
//            Ph[j][i] = C_w_h * h_w + C_ww_h * h_ww - C_e_h * h_e - C_ee_h * h_ee- C_s_h * h_s - C_ss_h * h_ss + C_n_h * h_n + C_nn_h * h_nn;
//
//
//            f_h[j][i] = Ph[j][i]/dt - (sin(phi_h[j][i])/r_z(phi_h[j][i]) * h[j][i]*(v[j][i]+v[j][i-1])/2.0);
//
//        }
//    }
//
//    //continuity eq integration
//
//    switch(n){
//
//        case 0:     //Euler
//
//            for(j=2; j<J-2; j++){
//                for(i=2; i<I-2; i++){
//
//                    eta[j][i] += dt * f_h[j][i];
//                    h[j][i] = D + eta[j][i];
//
//                    f_h_2old[j][i] = f_h[j][i];
//                }
//            }
//            break;
//
//        case 1:     //Euler
//
//            for(j=2; j<J-2; j++){
//                for(i=2; i<I-2; i++){
//
//                    eta[j][i] += dt * f_h[j][i];
//                    h[j][i] = D + eta[j][i];
//
//                    f_h_old[j][i] = f_h[j][i];
//                }
//            }
//            break;
//
//        default:        //Adam-Bashforth
//
//            for(j=2; j<J-2; j++){
//                for(i=2; i<I-2; i++){
//
//                    eta[j][i] += dt * (23.0/12.0 * f_h[j][i] - 4.0/3.0 * f_h_old[j][i] + 5.0/12.0 * f_h_2old[j][i]);
//                    h[j][i] = D + eta[j][i];
//
//                    f_h_2old[j][i] = f_h_old[j][i];
//                    f_h_old[j][i] = f_h[j][i];
//                }
//            }
//
//    }
//
//
//    //Canal Boundary conditions
//    for(j=0; j<J; j++){
//        //top
//        h[j][0] = 0.0;
//        h[j][1] = 0.0;
//
//        eta[j][0] = 0.0;
//        eta[j][1] = 0.0;
//
//        //bottom
//        h[j][I-1] = 0.0;
//        h[j][I-2] = 0.0;
//
//        eta[j][I-1] = 0.0;
//        eta[j][I-2] = 0.0;
//    }
//
//    for(i=0; i<I; i++){
//        //left
//        h[0][i] = h[J-4][i];
//        h[1][i] = h[J-3][i];
//
//        eta[0][i] = eta[J-4][i];
//        eta[1][i] = eta[J-3][i];
//
//        //right
//        h[J-1][i] = h[3][i];
//        h[J-2][i] = h[2][i];
//
//        eta[J-1][i] = eta[3][i];
//        eta[J-2][i] = eta[2][i];
//    }
//
//
//
//    //Potential vorticity
//
//    for(j=2; j<J-2; j++){
//        for(i=2; i<I-2; i++){
//
//            w[j][i] = (v[j+1][i] - v[j][i]) / dx_q[j][i] - (U[j][i+1] - U[j][i]) / dy_q[j][i];
//            q[j][i] = (f(phi_q[j][i]) + w[j][i]) / (D + (0.25 * (eta[j][i] + eta[j+1][i] + eta[j][i+1] + eta[j+1][i+1])));
//        }
//    }
//
//
//    //output in files
//
//    if(n%120==0){ //print 1 file every 1h
//        sprintf(buffer, "eta_%d_h.txt", n/720);  // numero 120: 120 iteracions per fer una h
//        output=fopen(buffer,"w");
//
//        for(j=2;j<J-2;j++){
//            for(i=2;i<I-2;i++){
//                fprintf(output, "%f \t %f \t %f \n", theta_h[j][i], phi_h[j][i], eta[j][i]);
//            }
//            fprintf(output, "\n");
//        }
//        fclose(output);
//
//
//        sprintf(buffer, "q_%d_h.txt", n/120);  // numero 120: 120 iteracions per fer una h
//        output=fopen(buffer,"w");
//
//        for(j=2;j<J-2;j++){
//            for(i=2;i<I-2;i++){
//                fprintf(output, "%f \t %f \t %.10f \n", theta_q[j][i], phi_q[j][i], q[j][i]);
//            }
//            fprintf(output, "\n");
//        }
//        fclose(output);
//
//
//        sprintf(buffer, "u_%d_h.txt", n/120);  // numero 120: 120 iteracions per fer una h
//        output=fopen(buffer,"w");
//
//        for(j=2;j<J-2;j++){
//            for(i=2;i<I-2;i++){
//                fprintf(output, "%f \t %f \t %f \n", theta_u[j][i], phi_u[j][i], U[j][i]);
//            }
//            fprintf(output, "\n");
//        }
//        fclose(output);
//
//
//        sprintf(buffer, "v_%d_h.txt", n/120);  // numero 120: 120 iteracions per fer una h
//        output=fopen(buffer,"w");
//
//        for(j=2;j<J-2;j++){
//            for(i=2;i<I-2;i++){
//                fprintf(output, "%f \t %f \t %f \n", theta_v[j][i], phi_v[j][i],v[j][i] );
//            }
//            fprintf(output, "\n");
//        }
//        fclose(output);
//    }
//
//
//
//
//}
    // Free memory
    free_mesh(&m);
    free_flow(&f); 
    return 0;
}
