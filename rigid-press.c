// Geometry optimization of molecular crystals in a regularized rigid-body approximation

// METHODOLOGY COMMENTS ON THIS OPTIMIZER:
// - It would be more efficient & robust to work with only the symmetry-inequivalent structural degrees of freedom,
//   but we don't have a convenient way of expressing those degrees of freedom, especially for Wyckoff positions.
//
// - It would be a little more efficient & stable to only consider 3 explicit degrees of freedom when perturbing
//   the orientational quaternions. For example, rather than an additive perturbation, a multiplicative perturbation
//   of the form (1, a, b, c) could be considered for infinitesimal a, b, & c.
//
// - There is nothing particularly special about the regularized rigid-body approximation that was chosen here,
//   it could be replaced by a more physical & accurate intermolecular interaction, which might improve the overlap
//   between the set of structures that are local minima here versus the true local minima of an accurate energy surface.
//
// - Space-group symmetries are likely to break if the initial structure is too loosely packed relative to the cutoff
//   distance of the regularized interatomic interaction. This can be mitigated by increasing the cutoff distance.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "crystal.h"
#include "cocrystal.h"
#include "rigid-press.h"

// prototypes for external dependencies (BLAS & LAPACK)
void dgemv_(char*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*);
void dgemm_(char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*);
void dsyev_(char*, char*, int*, double*, int*, double*, double*, int*, int*);
void dgeqrf_(int*, int*, double*, int*, double*, double*, int*, int*);
void dormqr_(char*, char*, int*, int*, int*, double*, int*, double*, double*, int*, double*, int*, int*);
void dgesvd_(char*, char*, int*, int*, double*, int*, double*, double*, int*, double*, int*, double*, int* ,int*);

// cutoff distance of the interaction kernel
#define INTERACTION_CUTOFF 10.0

// parameters defining the regularized interatomic contact interaction
#define INTERACTION_WEIGHT 0.1

// number of steps to take for each Golden-section line search
#define GOLDEN_STEPS 20

// optimization tolerance on energy change in 1 iteration relative to minimum energy
#define OPTIMIZATION_TOLERANCE 1e-6

// step size for numerical tests of analytical derivatives
#define STEP 1e-4

// the state vector of a crystal's geometry has size 6+7*nmol containing the following information:
// state[0] : 1st lattice vector (x)
// state[1-2] : 2nd lattice vector (x-y)
// state[3-5] : 3rd lattice vector (x-y-z)
// state[6-8] : center of 1st molecule (x-y-z)
// state[9-12] : orientational quaternion of 1st molecule
// ...

// information about a molecular crystal to define the structural relaxation problem
struct molecular_crystal
{
    // size information
    int ntype; // number of types of molecules in the crystal
    int nmol; // number of molecules per unit cell

    // information about each type of molecule
    int *natom; // number of atoms in a molecule type [ntype]
    double **geometry; // centered geometry of a molecule type (interleaved x-y-z format) [ntype][3*natom[i]]
    double ***collide; // collision distances for pairs of atoms between 2 molecules [ntype][ntype][natom[i]*natom[j]]

    // information about the crystal
    int *type; // type of each molecule in the unit cell [nmol]
    int *invert; // inversion of each molecule in the unit cell (+1 for standard, -1 for inverted) [nmol]
};

// deallocate memory in the molecular_crystal structure
void free_molecular_crystal(struct molecular_crystal *xtl)
{
    free(xtl->type);
    free(xtl->invert);

    for(int i=0 ; i<xtl->ntype ; i++)
    { free(xtl->geometry[i]); }
    free(xtl->geometry);

    for(int i=0 ; i<xtl->ntype ; i++)
    {
        for(int j=0 ; j<xtl->ntype ; j++)
        { free(xtl->collide[i][j]); }

        free(xtl->collide[i]);
    }
    free(xtl->collide);

    free(xtl->natom);
}

// interatomic interaction kernel & its 1st & 2nd distance derivatives
double kernel(double distance, // interatomic distance
              // kernel parameters: (add/replace parameters for more physical interactions)
              double distance0, // collision distance
              double wt) // kernel weight
{
    if(distance > INTERACTION_CUTOFF)
    { return 0.0; }
    if(distance < distance0)
    { return INFINITY; }
    return wt*(INTERACTION_CUTOFF - distance)/(distance - distance0);
}
double dkernel(double distance, // interatomic distance
              // kernel parameters: (add/replace parameters for more physical interactions)
              double distance0, // collision distance
              double wt) // kernel weight
{
    if(distance > INTERACTION_CUTOFF)
    { return 0.0; }
    if(distance < distance0)
    { return NAN; }
    double recip = 1.0/(distance - distance0);
    return -wt*((INTERACTION_CUTOFF - distance)*recip*recip + recip);
}
double d2kernel(double distance, // interatomic distance
              // kernel parameters: (add/replace parameters for more physical interactions)
              double distance0, // collision distance
              double wt) // kernel weight
{
    if(distance > INTERACTION_CUTOFF)
    { return 0.0; }
    if(distance < distance0)
    { return NAN; }
    double recip = 1.0/(distance - distance0);
    return 2.0*wt*((INTERACTION_CUTOFF - distance)*recip*recip*recip + recip*recip);
}

// position of atom in a translated & rotated molecule
void position(double *local, // local coordinate of atom in the molecule [3]
              double *state, // state vector of the molecule (x-y-z & orientational quaternion) [7]
              double *global) // global coordinate of the atom in the crystal [3]
{
    double wt = 2.0/(state[3]*state[3] + state[4]*state[4] + state[5]*state[5] + state[6]*state[6]);
    global[0] = state[0] + local[0] + wt*(-(state[5]*state[5] + state[6]*state[6])*local[0]
                                         + (state[4]*state[5] - state[3]*state[6])*local[1]
                                         + (state[4]*state[6] + state[3]*state[5])*local[2]);
    global[1] = state[1] + local[1] + wt*( (state[4]*state[5] + state[3]*state[6])*local[0]
                                         - (state[4]*state[4] + state[6]*state[6])*local[1]
                                         + (state[5]*state[6] - state[3]*state[4])*local[2]);
    global[2] = state[2] + local[2] + wt*( (state[4]*state[6] - state[3]*state[5])*local[0]
                                         + (state[5]*state[6] + state[3]*state[4])*local[1]
                                         - (state[4]*state[4] + state[5]*state[5])*local[2]);
}
// NOTE: simple derivatives w.r.t. state[0], state[1], & state[2] are ignored here
void position_derivative(double *local, // local coordinate of atom in the molecule [3]
                         double *state, // state vector of the molecule (x-y-z & orientational quaternion) [7]
                         double *global1, // global coordinate 1st derivatives of the atom in the crystal [12]
                         double *global2) // global coordinate 2nd derivatives of the atom in the crystal [48]
{
    double wt = 2.0/(state[3]*state[3] + state[4]*state[4] + state[5]*state[5] + state[6]*state[6]);
    double dwt = -wt*wt, dglobal[3], dglobal2[12];
    dglobal[0] = dwt*(-(state[5]*state[5] + state[6]*state[6])*local[0]
                     + (state[4]*state[5] - state[3]*state[6])*local[1]
                     + (state[4]*state[6] + state[3]*state[5])*local[2]);
    dglobal[1] = dwt*( (state[4]*state[5] + state[3]*state[6])*local[0]
                     - (state[4]*state[4] + state[6]*state[6])*local[1]
                     + (state[5]*state[6] - state[3]*state[4])*local[2]);
    dglobal[2] = dwt*( (state[4]*state[6] - state[3]*state[5])*local[0]
                     + (state[5]*state[6] + state[3]*state[4])*local[1]
                     - (state[4]*state[4] + state[5]*state[5])*local[2]);
    dglobal2[0 + 0*3] = dwt*(-state[6]*local[1] + state[5]*local[2]);
    dglobal2[0 + 1*3] = dwt*( state[5]*local[1] + state[6]*local[2]);
    dglobal2[0 + 2*3] = dwt*(-2.0*state[5]*local[0] + state[4]*local[1] + state[3]*local[2]);
    dglobal2[0 + 3*3] = dwt*(-2.0*state[6]*local[0] - state[3]*local[1] + state[4]*local[2]);

    dglobal2[1 + 0*3] = dwt*(state[6]*local[0] - state[4]*local[2]);
    dglobal2[1 + 1*3] = dwt*(state[5]*local[0] - 2.0*state[4]*local[1] - state[3]*local[2]);
    dglobal2[1 + 2*3] = dwt*(state[4]*local[0] + state[6]*local[2]);
    dglobal2[1 + 3*3] = dwt*(state[3]*local[0] - 2.0*state[6]*local[1] + state[5]*local[2]);

    dglobal2[2 + 0*3] = dwt*(-state[5]*local[0] + state[4]*local[1]);
    dglobal2[2 + 1*3] = dwt*( state[6]*local[0] + state[3]*local[1] - 2.0*state[4]*local[2]);
    dglobal2[2 + 2*3] = dwt*(-state[3]*local[0] + state[6]*local[1] - 2.0*state[5]*local[2]);
    dglobal2[2 + 3*3] = dwt*( state[4]*local[0] + state[5]*local[1]);

    for(int i=0 ; i<4 ; i++)
    for(int j=0 ; j<3 ; j++)
    { global1[j + i*3] = state[i+3]*dglobal[j]; }

    global1[0 + 0*3] += wt*(-state[6]*local[1] + state[5]*local[2]);
    global1[0 + 1*3] += wt*( state[5]*local[1] + state[6]*local[2]);
    global1[0 + 2*3] += wt*(-2.0*state[5]*local[0] + state[4]*local[1] + state[3]*local[2]);
    global1[0 + 3*3] += wt*(-2.0*state[6]*local[0] - state[3]*local[1] + state[4]*local[2]);

    global1[1 + 0*3] += wt*(state[6]*local[0] - state[4]*local[2]);
    global1[1 + 1*3] += wt*(state[5]*local[0] - 2.0*state[4]*local[1] - state[3]*local[2]);
    global1[1 + 2*3] += wt*(state[4]*local[0] + state[6]*local[2]);
    global1[1 + 3*3] += wt*(state[3]*local[0] - 2.0*state[6]*local[1] + state[5]*local[2]);

    global1[2 + 0*3] += wt*(-state[5]*local[0] + state[4]*local[1]);
    global1[2 + 1*3] += wt*( state[6]*local[0] + state[3]*local[1] - 2.0*state[4]*local[2]);
    global1[2 + 2*3] += wt*(-state[3]*local[0] + state[6]*local[1] - 2.0*state[5]*local[2]);
    global1[2 + 3*3] += wt*( state[4]*local[0] + state[5]*local[1]);

    for(int i=0 ; i<4 ; i++)
    for(int j=0 ; j<4 ; j++)
    for(int k=0 ; k<3 ; k++)
    {
        global2[k + j*3 + i*12] = -2.0*wt*state[i+3]*state[j+3]*dglobal[k]
                                + state[i+3]*dglobal2[k + j*3] + state[j+3]*dglobal2[k + i*3];
    }

    for(int i=0 ; i<4 ; i++)
    for(int j=0 ; j<3 ; j++)
    { global2[j + i*3 + i*12] += dglobal[j]; }

    global2[0 + 0*3 + 2*12] += wt*local[2];
    global2[0 + 0*3 + 3*12] += -wt*local[1];
    global2[0 + 1*3 + 2*12] += wt*local[1];
    global2[0 + 1*3 + 3*12] += wt*local[2];
    global2[0 + 2*3 + 0*12] += wt*local[2];
    global2[0 + 2*3 + 1*12] += wt*local[1];
    global2[0 + 2*3 + 2*12] += -2.0*wt*local[0];
    global2[0 + 3*3 + 0*12] += -wt*local[1];
    global2[0 + 3*3 + 1*12] += wt*local[2];
    global2[0 + 3*3 + 3*12] += -2.0*wt*local[0];

    global2[1 + 0*3 + 1*12] += -wt*local[2];
    global2[1 + 0*3 + 3*12] += wt*local[0];
    global2[1 + 1*3 + 0*12] += -wt*local[2];
    global2[1 + 1*3 + 1*12] += -2.0*wt*local[1];
    global2[1 + 1*3 + 2*12] += wt*local[0];
    global2[1 + 2*3 + 1*12] += wt*local[0];
    global2[1 + 2*3 + 3*12] += wt*local[2];
    global2[1 + 3*3 + 0*12] += wt*local[0];
    global2[1 + 3*3 + 2*12] += wt*local[2];
    global2[1 + 3*3 + 3*12] += -2.0*wt*local[1];

    global2[2 + 0*3 + 1*12] += wt*local[1];
    global2[2 + 0*3 + 2*12] += -wt*local[0];
    global2[2 + 1*3 + 0*12] += wt*local[1];
    global2[2 + 1*3 + 1*12] += -2.0*wt*local[2];
    global2[2 + 1*3 + 3*12] += wt*local[0];
    global2[2 + 2*3 + 0*12] += -wt*local[0];
    global2[2 + 2*3 + 2*12] += -2.0*wt*local[2];
    global2[2 + 2*3 + 3*12] += wt*local[1];
    global2[2 + 3*3 + 1*12] += wt*local[0];
    global2[2 + 3*3 + 2*12] += wt*local[1];
}
void position_derivative_test(double *local, // local coordinate of atom in the molecule [3]
                              double *state, // state vector of the molecule [7]
                              double *global1, // global coordinate 1st derivatives of the atom in the crystal [12]
                              double *global2) // global coordinate 2nd derivatives of the atom in the crystal [48]
{
    for(int i=0 ; i<4 ; i++)
    {
        double state0 = state[i+3], global_plus[3], global_minus[3], global1_plus[12], global1_minus[12], dummy[48];
        state[i+3] += STEP;
        position(local, state, global_plus);
        position_derivative(local, state, global1_plus, dummy);
        state[i+3] -= 2.0*STEP;
        position(local, state, global_minus);
        position_derivative(local, state, global1_minus, dummy);
        state[i+3] = state0;

        for(int j=0 ; j<3 ; j++)
        { global1[j + i*3] = (global_plus[j] - global_minus[j])/(2.0*STEP); }
        for(int j=0 ; j<12 ; j++)
        { global2[j + i*12] = (global1_plus[j] - global1_minus[j])/(2.0*STEP); }
    }
}

// interaction energy between a pair of molecules
double pair_energy(int natom1, // number of atoms in the 1st molecule
                   int natom2, // number of atoms in the 2nd molecule
                   int invert1, // inversion flag for 1st molecule {+1, -1}
                   int invert2, // inversion flag for 2nd molecule {+1, -1}
                   double *geo1, // geometry of the 1st molecule [3*natom1]
                   double *geo2, // geometry of the 2nd molecule [3*natom2]
                   double *collide, // collision matrix [natom1*natom2]
                   double *state1, // state vector of the 1st molecule [7]
                   double *state2) // state vector of the 2nd molecule [7]
{
    double energy = 0.0, wt = INTERACTION_WEIGHT/(double)(natom1*natom2);
    for(int i=0 ; i<natom1 ; i++)
    {
        double local1[3], coord1[3];
        local1[0] = invert1*geo1[3*i]; local1[1] = invert1*geo1[1+3*i]; local1[2] = invert1*geo1[2+3*i];
        position(local1, state1, coord1);

        for(int j=0 ; j<natom2 ; j++)
        {
            double local2[3], coord2[3];
            local2[0] = invert2*geo2[3*j]; local2[1] = invert2*geo2[1+3*j]; local2[2] = invert2*geo2[2+3*j];
            position(local2, state2, coord2);

            // regularized contact interaction
            double dist = sqrt((coord1[0] - coord2[0])*(coord1[0] - coord2[0])
                              +(coord1[1] - coord2[1])*(coord1[1] - coord2[1])
                              +(coord1[2] - coord2[2])*(coord1[2] - coord2[2]));
            energy += kernel(dist, collide[i+j*natom1], wt);
        }
    }
    return energy;
}
void pair_energy_derivative(int natom1, // number of atoms in the 1st molecule
                            int natom2, // number of atoms in the 2nd molecule
                            int invert1, // inversion flag for 1st molecule {+1, -1}
                            int invert2, // inversion flag for 2nd molecule {+1, -1}
                            double *geo1, // geometry of the 1st molecule [3*natom1]
                            double *geo2, // geometry of the 2nd molecule [3*natom2]
                            double *collide, // collision matrix [natom1*natom2]
                            double *state1, // state vector of the 1st molecule [7]
                            double *state2, // state vector of the 2nd molecule [7]
                            double *grad1, // 1st derivatives w.r.t. 1st state vector [7]
                            double *grad2, // 1st derivatives w.r.t. 2nd state vector [7]
                            double *hess11, // 2nd derivatives w.r.t. 1st state vector [49]
                            double *hess22, // 2nd derivatives w.r.t. 2nd state vector [49]
                            double *hess12) // mixed 2nd derivatives [49]
{
    for(int i=0 ; i<7 ; i++)
    { grad1[i] = grad2[i] = 0.0; }

    for(int i=0 ; i<49 ; i++)
    { hess11[i] = hess12[i] = hess22[i] = 0.0; }

    double wt = INTERACTION_WEIGHT/(double)(natom1*natom2);
    for(int i=0 ; i<natom1 ; i++)
    {
        double local1[3], coord1[3], coord1_deriv1[12], coord1_deriv2[48];
        local1[0] = invert1*geo1[3*i]; local1[1] = invert1*geo1[1+3*i]; local1[2] = invert1*geo1[2+3*i];
        position(local1, state1, coord1);
        position_derivative(local1, state1, coord1_deriv1, coord1_deriv2);

        for(int j=0 ; j<natom2 ; j++)
        {
            double local2[3], coord2[3], coord2_deriv1[12], coord2_deriv2[48];
            local2[0] = invert2*geo2[3*j]; local2[1] = invert2*geo2[1+3*j]; local2[2] = invert2*geo2[2+3*j];
            position(local2, state2, coord2);
            position_derivative(local2, state2, coord2_deriv1, coord2_deriv2);

            double dist = sqrt((coord1[0] - coord2[0])*(coord1[0] - coord2[0])
                              +(coord1[1] - coord2[1])*(coord1[1] - coord2[1])
                              +(coord1[2] - coord2[2])*(coord1[2] - coord2[2]));
            double denergy = dkernel(dist, collide[i+j*natom1], wt)/dist;
            double d2energy = (d2kernel(dist, collide[i+j*natom1], wt) - denergy)/(dist*dist);

            // simple derivatives w.r.t. state1[0], state1[1], state1[2], state2[0], state2[1], & state2[2]
            for(int k=0 ; k<3 ; k++)
            {
                double delta = coord1[k] - coord2[k];
                grad1[k] += denergy*delta;
                grad2[k] -= denergy*delta;
                hess11[k + k*7] += denergy;
                hess22[k + k*7] += denergy;
                hess12[k + k*7] -= denergy;
                for(int l=0 ; l<3 ; l++)
                {
                    double delta2 = coord1[l] - coord2[l];
                    hess11[l + k*7] += d2energy*delta*delta2;
                    hess22[l + k*7] += d2energy*delta*delta2;
                    hess12[l + k*7] -= d2energy*delta*delta2;
                }
            }

            // gradients & mixed quaternion-translation hessian
            for(int k=0 ; k<4 ; k++)
            for(int l=0 ; l<3 ; l++)
            {
                double delta = coord1[l] - coord2[l];
                grad1[k+3] += denergy*delta*coord1_deriv1[l+k*3];
                grad2[k+3] -= denergy*delta*coord2_deriv1[l+k*3];
                hess11[k+3 + l*7] += denergy*coord1_deriv1[l+k*3];
                hess22[k+3 + l*7] += denergy*coord2_deriv1[l+k*3];
                hess12[k+3 + l*7] -= denergy*coord1_deriv1[l+k*3];
                hess11[l + (k+3)*7] += denergy*coord1_deriv1[l+k*3];
                hess22[l + (k+3)*7] += denergy*coord2_deriv1[l+k*3];
                hess12[l + (k+3)*7] -= denergy*coord2_deriv1[l+k*3];

                for(int m=0 ; m<3 ; m++)
                {
                    double delta2 = coord1[m] - coord2[m];
                    hess11[k+3 + m*7] += d2energy*delta*delta2*coord1_deriv1[l+k*3];
                    hess22[k+3 + m*7] += d2energy*delta*delta2*coord2_deriv1[l+k*3];
                    hess12[k+3 + m*7] -= d2energy*delta*delta2*coord1_deriv1[l+k*3];
                    hess11[m + (k+3)*7] += d2energy*delta*delta2*coord1_deriv1[l+k*3];
                    hess22[m + (k+3)*7] += d2energy*delta*delta2*coord2_deriv1[l+k*3];
                    hess12[m + (k+3)*7] -= d2energy*delta*delta2*coord2_deriv1[l+k*3];
                }
            }

            // quaternion components of hessian
            for(int k=0 ; k<4 ; k++)
            for(int l=0 ; l<4 ; l++)
            for(int m=0 ; m<3 ; m++)
            {
                double delta = coord1[m] - coord2[m];
                hess11[k+3 + (l+3)*7] += denergy*coord1_deriv1[m+k*3]*coord1_deriv1[m+l*3];
                hess22[k+3 + (l+3)*7] += denergy*coord2_deriv1[m+k*3]*coord2_deriv1[m+l*3];
                hess12[k+3 + (l+3)*7] -= denergy*coord1_deriv1[m+k*3]*coord2_deriv1[m+l*3];
                hess11[k+3 + (l+3)*7] += denergy*delta*coord1_deriv2[m+k*3+l*12];
                hess22[k+3 + (l+3)*7] -= denergy*delta*coord2_deriv2[m+k*3+l*12];

                for(int n=0 ; n<3 ; n++)
                {
                    double delta2 = coord1[n] - coord2[n];
                    hess11[k+3 + (l+3)*7] += d2energy*delta*delta2*coord1_deriv1[m+k*3]*coord1_deriv1[n+l*3];
                    hess22[k+3 + (l+3)*7] += d2energy*delta*delta2*coord2_deriv1[m+k*3]*coord2_deriv1[n+l*3];
                    hess12[k+3 + (l+3)*7] -= d2energy*delta*delta2*coord1_deriv1[m+k*3]*coord2_deriv1[n+l*3];
                }
            }
        }
    }
}
void pair_energy_derivative_test(int natom1, // number of atoms in the 1st molecule
                                 int natom2, // number of atoms in the 2nd molecule
                                 int invert1, // inversion flag for 1st molecule {+1, -1}
                                 int invert2, // inversion flag for 2nd molecule {+1, -1}
                                 double *geo1, // geometry of the 1st molecule [3*natom1]
                                 double *geo2, // geometry of the 2nd molecule [3*natom2]
                                 double *collide, // collision matrix [natom1*natom2]
                                 double *state1, // state vector of the 1st molecule [7]
                                 double *state2, // state vector of the 2nd molecule [7]
                                 double *grad1, // 1st derivatives w.r.t. 1st state vector [7]
                                 double *grad2, // 1st derivatives w.r.t. 2nd state vector [7]
                                 double *hess11, // 2nd derivatives w.r.t. 1st state vector [49]
                                 double *hess22, // 2nd derivatives w.r.t. 2nd state vector [49]
                                 double *hess12) // mixed 2nd derivatives [49]
{
    for(int i=0 ; i<7; i++)
    {
        double state0 = state1[i], grad1_plus[7], grad1_minus[7], grad2_plus[7], grad2_minus[7], dummy[49];
        state1[i] += STEP;
        double energy_plus = pair_energy(natom1,natom2,invert1,invert2,geo1,geo2,collide,state1,state2);
        pair_energy_derivative(natom1,natom2,invert1,invert2,geo1,geo2,collide,state1,state2,grad1_plus,grad2_plus,dummy,dummy,dummy);
        state1[i] -= 2.0*STEP;
        double energy_minus = pair_energy(natom1,natom2,invert1,invert2,geo1,geo2,collide,state1,state2);
        pair_energy_derivative(natom1,natom2,invert1,invert2,geo1,geo2,collide,state1,state2,grad1_minus,grad2_minus,dummy,dummy,dummy);
        state1[i] = state0;

        grad1[i] = (energy_plus - energy_minus)/(2.0*STEP);
        for(int j=0 ; j<7 ; j++)
        {
            hess11[i+j*7] = (grad1_plus[j] - grad1_minus[j])/(2.0*STEP);
            hess12[i+j*7] = (grad2_plus[j] - grad2_minus[j])/(2.0*STEP);
        }

        state0 = state2[i];
        state2[i] += STEP;
        energy_plus = pair_energy(natom1,natom2,invert1,invert2,geo1,geo2,collide,state1,state2);
        pair_energy_derivative(natom1,natom2,invert1,invert2,geo1,geo2,collide,state1,state2,grad1_plus,grad2_plus,dummy,dummy,dummy);
        state2[i] -= 2.0*STEP;
        energy_minus = pair_energy(natom1,natom2,invert1,invert2,geo1,geo2,collide,state1,state2);
        pair_energy_derivative(natom1,natom2,invert1,invert2,geo1,geo2,collide,state1,state2,grad1_minus,grad2_minus,dummy,dummy,dummy);
        state2[i] = state0;

        grad2[i] = (energy_plus - energy_minus)/(2.0*STEP);
        for(int j=0 ; j<7 ; j++)
        { hess22[i+j*7] = (grad2_plus[j] - grad2_minus[j])/(2.0*STEP); }
    }
}

// calculate a lattice-aligned bounding box for a molecule
void bound_box(int natom, // number of atoms in the molecule
               int invert, // inversion of the molecule {+1, -1}
               double *geo, // atomic coordinates of the molecule [3*natom]
               double *state, // state of the molecule [7]
               double *reclat, // reciprocal lattice vectors [6]
               double *box) // bounding box [6]
{
    box[0] = box[2] = box[4] = INFINITY;
    box[1] = box[3] = box[5] = -INFINITY;

    for(int i=0 ; i<natom ; i++)
    {
        double local[3], global[3], latpos[3];
        local[0] = invert*geo[3*i]; local[1] = invert*geo[1+3*i]; local[2] = invert*geo[2+3*i];
        position(local, state, global);
        latpos[0] = reclat[0]*global[0] + reclat[1]*global[1] + reclat[2]*global[2];
        latpos[1] = reclat[3]*global[1] + reclat[4]*global[2];
        latpos[2] = reclat[5]*global[2];

        if(latpos[0] < box[0]) { box[0] = latpos[0]; }
        if(latpos[0] > box[1]) { box[1] = latpos[0]; }
        if(latpos[1] < box[2]) { box[2] = latpos[1]; }
        if(latpos[1] > box[3]) { box[3] = latpos[1]; }
        if(latpos[2] < box[4]) { box[4] = latpos[2]; }
        if(latpos[2] > box[5]) { box[5] = latpos[2]; }
    }
}


// energy function that we are minimizing to relax the molecular crystal
double total_energy(struct molecular_crystal *xtl, // description of the crystal being optimized
                    double *state) // the crystal's state vector [6+7*xtl->nmol]
{
    double energy = fabs(state[0]*state[2]*state[5]);

    // construct reciprocal lattice vectors (1st: x-y-z, 2nd: y-z, 3rd: z)
    double wt = 1.0/(state[0]*state[2]*state[5]), reclat[6];
    reclat[0] = wt*state[2]*state[5];
    reclat[1] = -wt*state[1]*state[5];
    reclat[2] = wt*(state[1]*state[4] - state[2]*state[3]);
    reclat[3] = wt*state[5]*state[0];
    reclat[4] = -wt*state[4]*state[0];
    reclat[5] = wt*state[0]*state[2];

    // calculate buffer (lattice-aligned bounding box for the interaction sphere)
    double buffer[3];
    buffer[0] = INTERACTION_CUTOFF/sqrt(state[0]*state[0]);
    buffer[1] = INTERACTION_CUTOFF/sqrt(state[1]*state[1] + state[2]*state[2]);
    buffer[2] = INTERACTION_CUTOFF/sqrt(state[3]*state[3] + state[4]*state[4] + state[5]*state[5]);

    // loop over molecules in the central unit cell
    for(int i=0 ; i<xtl->nmol ; i++)
    {
        double *state1 = state + 6 + 7*i;

        // calculate lattice-aligned bounding box for molecule 1
        double box1[6];
        bound_box(xtl->natom[xtl->type[i]], xtl->invert[i], xtl->geometry[xtl->type[i]], state1, reclat, box1);

        // loop over molecules in 2nd unit cell
        for(int j=0 ; j<xtl->nmol ; j++)
        {
            // calculate lattice-aligned bounding box for molecule 2
            double box2[6];
            bound_box(xtl->natom[xtl->type[j]], xtl->invert[j], xtl->geometry[xtl->type[j]], state + 6 + 7*j, reclat, box2);

            // adjust lattice vector summation for pair of molecules
            int latmin1, latmin2, latmin3, latmax1, latmax2, latmax3;
            latmin1 = floor(box1[0] - box2[1] - buffer[0]);
            latmin2 = floor(box1[2] - box2[3] - buffer[1]);
            latmin3 = floor(box1[4] - box2[5] - buffer[2]);
            latmax1 = ceil(box1[1] - box2[0] + buffer[0]);
            latmax2 = ceil(box1[3] - box2[2] + buffer[1]);
            latmax3 = ceil(box1[5] - box2[4] + buffer[2]);

            // loop over interacting unit cells
            for(int k=latmin1 ; k<=latmax1 ; k++)
            for(int l=latmin2 ; l<=latmax2 ; l++)
            for(int m=latmin3 ; m<=latmax3 ; m++)
            {
                // molecules don't interact with themselves & only consider unique pairs in central cell
                if(i >= j && k == 0 && l == 0 && m == 0)
                { continue; }

                // shift center of the 2nd molecule
                double state2[7];
                for(int n=0 ; n<7 ; n++)
                { state2[n] = state[n+6+7*j]; }
                state2[0] += k*state[0] + l*state[1] + m*state[3];
                state2[1] += l*state[2] + m*state[4];
                state2[2] += m*state[5];

                energy += pair_energy(xtl->natom[xtl->type[i]], xtl->natom[xtl->type[j]],
                                      xtl->invert[i], xtl->invert[j],
                                      xtl->geometry[xtl->type[i]], xtl->geometry[xtl->type[j]],
                                      xtl->collide[xtl->type[i]][xtl->type[j]],
                                      state1, state2);
                if(energy == INFINITY)
                { return energy; }
            }
        }
    }
    return energy;
}
void total_energy_derivative(struct molecular_crystal *xtl, // description of the crystal being optimized
                             double *state, // the crystal's state vector [6+7*xtl->nmol]
                             double *grad, // 1st derivative of the total energy [6+7*xtl->nmol]
                             double *hess) // 2nd derivative of the total energy [(6+7*xtl->nmol)*(6+7*xtl->nmol)]
{
    int size = 6+7*xtl->nmol;
    double sign0 = 1.0, sign2 = 1.0, sign5 = 1.0;
    if(state[0] < 0.0)
    { sign0 = -1.0; }
    if(state[2] < 0.0)
    { sign2 = -1.0; }
    if(state[5] < 0.0)
    { sign5 = -1.0; }

    for(int i=0 ; i<size ; i++)
    { grad[i] = 0.0; }
    grad[0] = fabs(state[2]*state[5])*sign0;
    grad[2] = fabs(state[0]*state[5])*sign2;
    grad[5] = fabs(state[0]*state[2])*sign5;

    for(int i=0 ; i<size*size ; i++)
    { hess[i] = 0.0; }
    hess[0 + 2*size] = fabs(state[5])*sign0*sign2;
    hess[2 + 0*size] = fabs(state[5])*sign0*sign2;
    hess[0 + 5*size] = fabs(state[2])*sign0*sign5;
    hess[5 + 0*size] = fabs(state[2])*sign0*sign5;
    hess[2 + 5*size] = fabs(state[0])*sign2*sign5;
    hess[5 + 2*size] = fabs(state[0])*sign2*sign5;

    // construct reciprocal lattice vectors (1st: x-y-z, 2nd: y-z, 3rd: z)
    double wt = 1.0/(state[0]*state[2]*state[5]), reclat[6];
    reclat[0] = wt*state[2]*state[5];
    reclat[1] = -wt*state[1]*state[5];
    reclat[2] = wt*(state[1]*state[4] - state[2]*state[3]);
    reclat[3] = wt*state[5]*state[0];
    reclat[4] = -wt*state[4]*state[0];
    reclat[5] = wt*state[0]*state[2];

    // calculate buffer (lattice-aligned bounding box for the interaction sphere)
    double buffer[3];
    buffer[0] = INTERACTION_CUTOFF/sqrt(state[0]*state[0]);
    buffer[1] = INTERACTION_CUTOFF/sqrt(state[1]*state[1] + state[2]*state[2]);
    buffer[2] = INTERACTION_CUTOFF/sqrt(state[3]*state[3] + state[4]*state[4] + state[5]*state[5]);

    // loop over molecules in the central unit cell
    for(int i=0 ; i<xtl->nmol ; i++)
    {
        double *state1 = state + 6 + 7*i;

        // calculate lattice-aligned bounding box for molecule 1
        double box1[6];
        bound_box(xtl->natom[xtl->type[i]], xtl->invert[i], xtl->geometry[xtl->type[i]], state1, reclat, box1);

        // loop over molecules in 2nd unit cell
        for(int j=0 ; j<xtl->nmol ; j++)
        {
            // calculate lattice-aligned bounding box for molecule 2
            double box2[6];
            bound_box(xtl->natom[xtl->type[j]], xtl->invert[j], xtl->geometry[xtl->type[j]], state + 6 + 7*j, reclat, box2);

            // adjust lattice vector summation for pair of molecules
            int latmin1, latmin2, latmin3, latmax1, latmax2, latmax3;
            latmin1 = floor(box1[0] - box2[1] - buffer[0]);
            latmin2 = floor(box1[2] - box2[3] - buffer[1]);
            latmin3 = floor(box1[4] - box2[5] - buffer[2]);
            latmax1 = ceil(box1[1] - box2[0] + buffer[0]);
            latmax2 = ceil(box1[3] - box2[2] + buffer[1]);
            latmax3 = ceil(box1[5] - box2[4] + buffer[2]);

            // loop over interacting unit cells
            for(int k=latmin1 ; k<=latmax1 ; k++)
            for(int l=latmin2 ; l<=latmax2 ; l++)
            for(int m=latmin3 ; m<=latmax3 ; m++)
            {
                // molecules don't interact with themselves & only consider unique pairs in central cell
                if(i >= j && k == 0 && l == 0 && m == 0)
                { continue; }

                // shift center of the 2nd molecule
                double state2[7];
                for(int n=0 ; n<7 ; n++)
                { state2[n] = state[n+6+7*j]; }
                state2[0] += k*state[0] + l*state[1] + m*state[3];
                state2[1] += l*state[2] + m*state[4];
                state2[2] += m*state[5];

                double grad1[7], grad2[7], hess11[49], hess22[49], hess12[49];
                pair_energy_derivative(xtl->natom[xtl->type[i]], xtl->natom[xtl->type[j]],
                                       xtl->invert[i], xtl->invert[j],
                                       xtl->geometry[xtl->type[i]], xtl->geometry[xtl->type[j]],
                                       xtl->collide[xtl->type[i]][xtl->type[j]],
                                       state1, state2, grad1, grad2, hess11, hess22, hess12);
                for(int n=0 ; n<7 ; n++)
                {
                    grad[n+6+i*7] += grad1[n];
                    grad[n+6+j*7] += grad2[n];
                }
                grad[0] += k*grad2[0];
                grad[1] += l*grad2[0];
                grad[2] += l*grad2[1];
                grad[3] += m*grad2[0];
                grad[4] += m*grad2[1];
                grad[5] += m*grad2[2];

                for(int n=0 ; n<7 ; n++)
                for(int o=0 ; o<7 ; o++)
                {
                    hess[n+6+i*7 + (o+6+i*7)*size] += hess11[n + o*7];
                    hess[n+6+j*7 + (o+6+j*7)*size] += hess22[n + o*7];
                    hess[n+6+i*7 + (o+6+j*7)*size] += hess12[n + o*7];
                    hess[n+6+j*7 + (o+6+i*7)*size] += hess12[o + n*7];
                }
                for(int n=0 ; n<7 ; n++)
                {
                    hess[0 + (n+6+i*7)*size] += k*hess12[n + 0*7];
                    hess[1 + (n+6+i*7)*size] += l*hess12[n + 0*7];
                    hess[2 + (n+6+i*7)*size] += l*hess12[n + 1*7];
                    hess[3 + (n+6+i*7)*size] += m*hess12[n + 0*7];
                    hess[4 + (n+6+i*7)*size] += m*hess12[n + 1*7];
                    hess[5 + (n+6+i*7)*size] += m*hess12[n + 2*7];
                    hess[0 + (n+6+j*7)*size] += k*hess22[n + 0*7];
                    hess[1 + (n+6+j*7)*size] += l*hess22[n + 0*7];
                    hess[2 + (n+6+j*7)*size] += l*hess22[n + 1*7];
                    hess[3 + (n+6+j*7)*size] += m*hess22[n + 0*7];
                    hess[4 + (n+6+j*7)*size] += m*hess22[n + 1*7];
                    hess[5 + (n+6+j*7)*size] += m*hess22[n + 2*7];

                    hess[n+6+i*7 + 0*size] += k*hess12[n + 0*7];
                    hess[n+6+i*7 + 1*size] += l*hess12[n + 0*7];
                    hess[n+6+i*7 + 2*size] += l*hess12[n + 1*7];
                    hess[n+6+i*7 + 3*size] += m*hess12[n + 0*7];
                    hess[n+6+i*7 + 4*size] += m*hess12[n + 1*7];
                    hess[n+6+i*7 + 5*size] += m*hess12[n + 2*7];
                    hess[n+6+j*7 + 0*size] += k*hess22[n + 0*7];
                    hess[n+6+j*7 + 1*size] += l*hess22[n + 0*7];
                    hess[n+6+j*7 + 2*size] += l*hess22[n + 1*7];
                    hess[n+6+j*7 + 3*size] += m*hess22[n + 0*7];
                    hess[n+6+j*7 + 4*size] += m*hess22[n + 1*7];
                    hess[n+6+j*7 + 5*size] += m*hess22[n + 2*7];
                }
                hess[0 + 0*size] += k*k*hess22[0 + 0*7];
                hess[1 + 0*size] += k*l*hess22[0 + 0*7];
                hess[2 + 0*size] += k*l*hess22[1 + 0*7];
                hess[3 + 0*size] += k*m*hess22[0 + 0*7];
                hess[4 + 0*size] += k*m*hess22[1 + 0*7];
                hess[5 + 0*size] += k*m*hess22[2 + 0*7];
                hess[0 + 1*size] += l*k*hess22[0 + 0*7];
                hess[1 + 1*size] += l*l*hess22[0 + 0*7];
                hess[2 + 1*size] += l*l*hess22[1 + 0*7];
                hess[3 + 1*size] += l*m*hess22[0 + 0*7];
                hess[4 + 1*size] += l*m*hess22[1 + 0*7];
                hess[5 + 1*size] += l*m*hess22[2 + 0*7];
                hess[0 + 2*size] += l*k*hess22[0 + 1*7];
                hess[1 + 2*size] += l*l*hess22[0 + 1*7];
                hess[2 + 2*size] += l*l*hess22[1 + 1*7];
                hess[3 + 2*size] += l*m*hess22[0 + 1*7];
                hess[4 + 2*size] += l*m*hess22[1 + 1*7];
                hess[5 + 2*size] += l*m*hess22[2 + 1*7];
                hess[0 + 3*size] += m*k*hess22[0 + 0*7];
                hess[1 + 3*size] += m*l*hess22[0 + 0*7];
                hess[2 + 3*size] += m*l*hess22[1 + 0*7];
                hess[3 + 3*size] += m*m*hess22[0 + 0*7];
                hess[4 + 3*size] += m*m*hess22[1 + 0*7];
                hess[5 + 3*size] += m*m*hess22[2 + 0*7];
                hess[0 + 4*size] += m*k*hess22[0 + 1*7];
                hess[1 + 4*size] += m*l*hess22[0 + 1*7];
                hess[2 + 4*size] += m*l*hess22[1 + 1*7];
                hess[3 + 4*size] += m*m*hess22[0 + 1*7];
                hess[4 + 4*size] += m*m*hess22[1 + 1*7];
                hess[5 + 4*size] += m*m*hess22[2 + 1*7];
                hess[0 + 5*size] += m*k*hess22[0 + 2*7];
                hess[1 + 5*size] += m*l*hess22[0 + 2*7];
                hess[2 + 5*size] += m*l*hess22[1 + 2*7];
                hess[3 + 5*size] += m*m*hess22[0 + 2*7];
                hess[4 + 5*size] += m*m*hess22[1 + 2*7];
                hess[5 + 5*size] += m*m*hess22[2 + 2*7];

                if(isnan(grad1[0]))
                {
                    for(int n=0 ; n<size ; n++)
                    { grad[n] = NAN; }
                    return;
                }
            }
        }
    }
}
// numerical derivatives for debugging purposes
void total_energy_derivative_test(struct molecular_crystal *xtl, // description of the crystal being optimized
                                  double *state, // the crystal's state vector [6+7*xtl->nmol]
                                  double *grad, // 1st derivative of the total energy [6+7*xtl->nmol]
                                  double *hess) // 2nd derivative of the total energy [(6+7*xtl->nmol)*(6+7*xtl->nmol)]
{
    int num = 6+7*xtl->nmol;
    double *grad_plus = (double*)malloc(sizeof(double)*num);
    double *grad_minus = (double*)malloc(sizeof(double)*num);
    double *dummy = (double*)malloc(sizeof(double)*num*num);
    for(int i=0 ; i<num ; i++)
    {
        double state0 = state[i];
        state[i] += STEP;
        double energy_plus = total_energy(xtl,state);
        total_energy_derivative(xtl,state,grad_plus,dummy);
        state[i] -= 2.0*STEP;
        double energy_minus = total_energy(xtl,state);
        total_energy_derivative(xtl,state,grad_minus,dummy);
        state[i] = state0;

        grad[i] = (energy_plus - energy_minus)/(2.0*STEP);

        for(int j=0 ; j<num ; j++)
        { hess[i+j*num] = (grad_plus[j] - grad_minus[j])/(2.0*STEP); }
    }
    free(grad_plus);
    free(grad_minus);
    free(dummy);
}

// renormalize quaternions
void renormalize(int nmol, // number of molecules in the state vector
                 double *state) // state vector [6+7*nmol]
{
    for(int i=0 ; i<nmol ; i++)
    {
        double renorm = 1.0/sqrt(state[9+7*i]*state[9+7*i] + state[10+7*i]*state[10+7*i]
                                + state[11+7*i]*state[11+7*i] + state[12+7*i]*state[12+7*i]);
        for(int j=0 ; j<4 ; j++)
        { state[9+j+7*i] *= renorm; }
    }
}

// objective function to optimize the volume of the molecular crystal
double volume_search(double x, // optimization variable [0,1]
                     struct molecular_crystal *xtl, // description of the crystal being optimized
                     double *state, // the crystal's state vector [6+7*xtl->nmol]
                     double *min, // smallest scale factor to be considered [1]
                     double *max, // largest scale factor to be considered [1]
                     double *dummy, // dummy variable to match argument list w/ other objective function
                     double *work) // work vector [6+7*xtl->nmol]
{
    int size = 6+7*xtl->nmol;
    double scale0 = (1.0-x)*min[0] + x*max[0];
    for(int i=0 ; i<size ; i++)
    { work[i] = scale0*state[i]; }
    return total_energy(xtl, work);
}

// objective function for the Tikhonov-regularized line search
double quad_search(double x, // optimization variable [0,1]
                   struct molecular_crystal *xtl, // description of the crystal being optimized
                   double *state, // the crystal's state vector in normal coordinates [6+7*xtl->nmol]
                   double *grad, // gradient in normal coordinates [6+7*xtl->nmol]
                   double *eval, // eigenvalues of the Hessian matrix [6+7*xtl->nmol]
                   double *evec, // eigenvectors of the Hessian matrix [(6+7*xtl->nmol)*(6+7*xtl->nmol)]
                   double *work) // work vector [13+14*xtl->nmol]
{
    int size = 6+7*xtl->nmol;
    x *= work[2*size]; // hack to tune the search interval
    for(int i=0 ; i<size ; i++)
    {
        work[i] = state[i];
        work[i+size] = -x*grad[i]/(fabs(x*eval[i]) + (1-x)*fabs(eval[size-1]));
    }
    char notrans = 'N';
    int inc = 1;
    double one = 1.0;
    dgemv_(&notrans, &size, &size, &one, evec, &size, work+size, &inc, &one, work, &inc);
    return total_energy(xtl, work);
}

// line optimizer for both objective functions (Golden-section search)
// code adapted from a Python implementation at https://en.wikipedia.org/wiki/Golden-section_search
double line_optimize(struct molecular_crystal *xtl, // description of the crystal being optimized
                     int nstep, // number of optimization steps
                     double *state, // the crystal's state vector in normal coordinates [6+7*xtl->nmol]
                     double (*fptr)(double, struct molecular_crystal*, double*, double*, double*, double*, double*),
                     double *vec1,
                     double *vec2,
                     double *vec3,
                     double *vec4)
{
    double invphi = (sqrt(5.0) - 1.0)*0.5, invphi2 = (3.0 - sqrt(5.0))*0.5;
    double a = 0.0, b = 1.0, c = invphi2, d = invphi, h = 1.0;
    double yc = fptr(c, xtl, state, vec1, vec2, vec3, vec4);
    double yd = fptr(d, xtl, state, vec1, vec2, vec3, vec4);

    for(int i=0 ; i<nstep ; i++)
    {
        if(yc < yd)
        {
            b = d;
            d = c;
            yd = yc;
            h *= invphi;
            c = a + invphi2*h;
            yc = fptr(c, xtl, state, vec1, vec2, vec3, vec4);
        }
        else
        {
            a = c;
            c = d;
            yc = yd;
            h *= invphi;
            d = a + invphi*h;
            yd = fptr(d, xtl, state, vec1, vec2, vec3, vec4);
        }
    }

    // crappy hack to load the minimizer into the workspace (extra computation of objective function)
    double min = c, ymin = yc;
    if(yd < yc) { min = d; ymin = yd; }
    fptr(min, xtl, state, vec1, vec2, vec3, vec4);

    // extract minimizer from the workspace (a hack) & renormalize quaternions
    int size = 6+7*xtl->nmol;
    for(int i=0 ; i<size ; i++)
    { state[i] = vec4[i]; }
    renormalize(xtl->nmol, state);
    return ymin;
}

// main loop of crystal optimization
void optimize(struct molecular_crystal *xtl, // description of the crystal being optimized
              double *state, // the crystal's state vector to be updated [6+7*xtl->nmol]
              int family) // crystal family (see key in rigid-press.h)
{
    int size = 6+7*xtl->nmol;
    double *workspace = (double*)malloc(sizeof(double)*size*2);

    // construct a constraint matrix for high-symmetry lattice vectors
    double constraint_mat[36];
    for(int i=0 ; i<36 ; i++)
    { constraint_mat[i] = 0.0; }
    switch(family)
    {
        case 1:
        constraint_mat[0 + 0*6] = constraint_mat[2 + 2*6] = constraint_mat[3 + 3*6] = constraint_mat[5 + 5*6] = 1.0;
        break;

        case 2:
        constraint_mat[0 + 0*6] = constraint_mat[2 + 2*6] = constraint_mat[5 + 5*6] = 1.0;
        break;

        case 3:
        constraint_mat[5 + 5*6] = 1.0;
        constraint_mat[0 + 0*6] = constraint_mat[2 + 0*6] = constraint_mat[0 + 2*6] = constraint_mat[2 + 2*6] = 0.5;
        break;

        case 4:
        constraint_mat[5 + 5*6] = 1.0;
        constraint_mat[0 + 0*6] = 0.5;
        constraint_mat[1 + 1*6] = 0.125;
        constraint_mat[2 + 2*6] = 0.375;
        constraint_mat[0 + 1*6] = constraint_mat[1 + 0*6] = -0.25;
        constraint_mat[0 + 2*6] = constraint_mat[2 + 0*6] = sqrt(3.0)*0.25;
        constraint_mat[1 + 2*6] = constraint_mat[2 + 1*6] = -sqrt(3.0)*0.125;
        break;

        case 5:
        constraint_mat[0 + 0*6] = constraint_mat[2 + 0*6] = constraint_mat[5 + 0*6] = 1.0/3.0;
        constraint_mat[0 + 2*6] = constraint_mat[2 + 2*6] = constraint_mat[5 + 2*6] = 1.0/3.0;
        constraint_mat[0 + 5*6] = constraint_mat[2 + 5*6] = constraint_mat[5 + 5*6] = 1.0/3.0;
        break;
    }

    // find an overpacked volume
    double scale_min = 1.0, scale_max = 1.0;
    double energy = total_energy(xtl, state), energy_min = energy, energy_max = energy;
    while(energy_min != INFINITY)
    {
        scale_min *= 0.5;
        energy_min = volume_search(0.0, xtl, state, &scale_min, &scale_max, NULL, workspace);
    }

    // find an underpacked volume
    while(energy_max == INFINITY)
    {
        scale_max *= 2.0;
        energy_max = volume_search(1.0, xtl, state, &scale_min, &scale_max, NULL, workspace);
        if(energy_max == INFINITY)
        { scale_min = scale_max; }
    }
    do
    {
        energy = energy_max;
        scale_max *= 2.0;
        energy_max = volume_search(1.0, xtl, state, &scale_min, &scale_max, NULL, workspace);
    } while(energy > energy_max);

    // preliminary volume optimization
    energy = line_optimize(xtl, GOLDEN_STEPS, state, volume_search, &scale_min, &scale_max, NULL, workspace);

    // main optimization loop
    int iter = 0, lwork = -1, info, inc = 1, progress = 1, six = 6;
    char jobz = 'V', uplo = 'U', notrans = 'N', trans = 'T';
    double work0, one = 1.0, zero = 0.0;
    double *grad = (double*)malloc(sizeof(double)*size);
    double *hess = (double*)malloc(sizeof(double)*size*size);
    double *ev = (double*)malloc(sizeof(double)*size);
    dsyev_(&jobz, &uplo, &size, NULL, &size, NULL, &work0, &lwork, &info);
    lwork = (int)work0;
    if(lwork < 6*size)
    { lwork = 6*size; }
    double *work = (double*)malloc(sizeof(double)*lwork);
    double new_energy = energy;
    do
    {
        // save previous energy
        energy = new_energy;

        // expand the total energy to 2nd order
        total_energy_derivative(xtl, state, grad, hess);

        // apply constraints to gradient & Hessian
        if(family != 0)
        {
            dgemv_(&notrans, &six, &six, &one, constraint_mat, &six, grad, &inc, &zero, work, &inc);
            for(int i=0 ; i<6 ; i++)
            { grad[i] = work[i]; }

            dgemm_(&notrans, &notrans, &six, &size, &six, &one, constraint_mat, &six, hess, &size, &zero, work, &six);
            for(int i=0 ; i<6 ; i++)
            for(int j=0 ; j<size ; j++)
            { hess[i+j*size] = work[i+j*6]; }

            dgemm_(&notrans, &notrans, &size, &six, &six, &one, hess, &size, constraint_mat, &six, &zero, work, &size);
            for(int i=0 ; i<size ; i++)
            for(int j=0 ; j<6 ; j++)
            { hess[i+j*size] = work[i+j*size]; }
        }

        // transform into normal modes of the quadratic approximant
        dsyev_(&jobz, &uplo, &size, hess, &size, ev, work, &lwork, &info);
        for(int i=0 ; i<size ; i++)
        { work[i] = grad[i]; }
        dgemv_(&trans, &size, &size, &one, hess, &size, work, &inc, &zero, grad, &inc);

        // identify a reasonable search interval
        double width = 1.0;
        workspace[2*size] = 1.0; // quick hack to inject a tunable search interval into quad_search
        while(quad_search((sqrt(5.0) - 1.0)*0.5*width, xtl, state, grad, ev, hess, workspace) > energy)
        { width *= (sqrt(5.0) - 1.0)*0.5; }
        workspace[2*size] = width; // quick hack to inject a tunable search interval into quad_search

        // perform a Tikhonov-regularized line search
        new_energy = line_optimize(xtl, GOLDEN_STEPS, state, quad_search, grad, ev, hess, workspace);

        // strictly enforce constraints at the end of every optimization step
        switch(family)
        {
            case 1:
            state[1] = state[4] = 0.0;
            break;

            case 2:
            state[1] = state[3] = state[4] = 0.0;
            break;

            case 3:
            state[1] = state[3] = state[4] = 0.0;
            state[2] = state[0];
            break;

            case 4:
            state[3] = state[4] = 0.0;
            state[1] = -0.5*state[0];
            state[2] = sqrt(3.0)*0.5*state[0];
            break;

            case 5:
            state[1] = state[3] = state[4] = 0.0;
            state[2] = state[0];
            state[5] = state[0];
            break;
        }
    } while((energy - new_energy) > OPTIMIZATION_TOLERANCE*fabs(new_energy));

    free(workspace);
    free(grad);
    free(hess);
    free(ev);
    free(work);
}

// adapted from pseudocode in "Converting a Rotation Matrix to a Quaternion" by Mike Day
void matrix2quaternion(double *rot, double *quat)
{
    double t;
    if (rot[2+2*3] < 0.0)
    {
        if (rot[0+0*3] > rot[1+1*3])
        {
            t = 1.0 + rot[0+0*3] - rot[1+1*3] - rot[2+2*3];
            quat[0] = rot[2+1*3]-rot[1+2*3];
            quat[1] = t;
            quat[2] = rot[1+0*3]+rot[0+1*3];
            quat[3] = rot[2+0*3]+rot[0+2*3];
        }
        else
        {
            t = 1.0 - rot[0+0*3] + rot[1+1*3] - rot[2+2*3];
            quat[0] = rot[0+2*3]-rot[2+0*3];
            quat[1] = rot[1+0*3]+rot[0+1*3];
            quat[2] = t;
            quat[3] = rot[2+1*3]+rot[1+2*3];
        }
    }
    else
    {
        if (rot[0+0*3] < -rot[1+1*3])
        {
            t = 1.0 - rot[0+0*3] - rot[1+1*3] + rot[2+2*3];
            quat[0] = rot[1+0*3]-rot[0+1*3];
            quat[1] = rot[0+2*3]+rot[2+0*3];
            quat[2] = rot[2+1*3]+rot[1+2*3];
            quat[3] = t;
        }
        else
        {
            t = 1.0 + rot[0+0*3] + rot[1+1*3] + rot[2+2*3];
            quat[0] = t;
            quat[1] = rot[2+1*3]-rot[1+2*3];
            quat[2] = rot[0+2*3]-rot[2+0*3];
            quat[3] = rot[1+0*3]-rot[0+1*3];
        }
    }
    t = 0.5/sqrt(t);
    quat[0] *= t;
    quat[1] *= t;
    quat[2] *= t;
    quat[3] *= t;
}

// NOTE: rows/columns of the cutoff_matrix are over all atoms in the unit cell
void optimize_crystal(crystal *xtl, double *cutoff_matrix, int family)
{
/*
printf("initial geometry:\n");
for(int i=0 ; i<3 ; i++)
{
    printf("latt %d %f %f %f\n", i, xtl->lattice_vectors[i][0], xtl->lattice_vectors[i][1], xtl->lattice_vectors[i][2]);
}
for(int j=0 ; j<xtl->Z*xtl->num_atoms_in_molecule ; j++)
{
    printf("pos %d %f %f %f\n", j, xtl->Xcord[j], xtl->Ycord[j], xtl->Zcord[j]);
}
*/
    // test of lattice vector formatting
    if(family != 0 && (xtl->lattice_vectors[0][1] != 0.0 || xtl->lattice_vectors[0][2] != 0.0 || xtl->lattice_vectors[1][2] != 0.0))
    { printf("ERROR: incorrect lattice vector format in optimize_crystal"); exit(1); }

    // allocate memory for the temporary crystal data structure & state vector
    struct molecular_crystal xtl2;
    xtl2.ntype = 1;
    xtl2.nmol = xtl->Z;
    xtl2.natom = (int*)malloc(sizeof(int)*1);
    xtl2.natom[0] = xtl->num_atoms_in_molecule;
    xtl2.geometry = (double**)malloc(sizeof(double*)*1);
    xtl2.geometry[0] = (double*)malloc(sizeof(double)*3*xtl2.natom[0]);
    xtl2.collide = (double***)malloc(sizeof(double**)*1);
    xtl2.collide[0] = (double**)malloc(sizeof(double*)*1);
    xtl2.collide[0][0] = (double*)malloc(sizeof(double)*xtl2.natom[0]*xtl2.natom[0]);
    for(int i=0 ; i<xtl2.natom[0] ; i++)
    for(int j=0 ; j<xtl2.natom[0] ; j++)
    { xtl2.collide[0][0][j+i*xtl2.natom[0]] = cutoff_matrix[j+i*xtl2.natom[0]*xtl2.nmol]; }
    xtl2.type = (int*)malloc(sizeof(int)*xtl2.nmol);
    xtl2.invert = (int*)malloc(sizeof(int)*xtl2.nmol);
    for(int i=0 ; i<xtl2.nmol ; i++)
    { xtl2.type[i] = 0; xtl2.invert[i] = 1; }
    double *state = (double*)malloc(sizeof(double)*(6 + 7*xtl2.nmol));

    // form rotation matrix to align lattice vectors (QR decomposition)
    double latvec[9], latvec2[9];
    for(int i=0 ; i<3 ; i++)
    for(int j=0 ; j<3 ; j++)
    { latvec[j + i*3] = xtl->lattice_vectors[i][j]; }
    for(int i=0 ; i<9 ; i++)
    { latvec2[i] = latvec[i]; }
    double tau[3];
    int lwork = xtl2.natom[0]+100, info, three = 3, one = 1;
    double *work = (double*)malloc(sizeof(double)*lwork);
    if(family == 0)
    { dgeqrf_(&three, &three, latvec2, &three, tau, work, &lwork, &info); }
    state[0] = latvec2[0 + 0*3];
    state[1] = latvec2[0 + 1*3];
    state[2] = latvec2[1 + 1*3];
    state[3] = latvec2[0 + 2*3];
    state[4] = latvec2[1 + 2*3];
    state[5] = latvec2[2 + 2*3];

    // isolate reference molecule
    char trans = 'T', notrans = 'N', left = 'L';
    for(int i=0 ; i<xtl2.natom[0] ; i++)
    {
        xtl2.geometry[0][0+3*i] = xtl->Xcord[i];
        xtl2.geometry[0][1+3*i] = xtl->Ycord[i];
        xtl2.geometry[0][2+3*i] = xtl->Zcord[i];
    }
    if(family == 0)
    { dormqr_(&left, &trans, &three, xtl2.natom, &three, latvec2, &three, tau, xtl2.geometry[0], &three, work, &lwork, &info); }

    // extract geometric center & center reference geometry
    double coord[3];
    coord[0] = coord[1] = coord[2] = 0.0;
    for(int i=0 ; i<xtl2.natom[0] ; i++)
    for(int j=0 ; j<3 ; j++)
    { coord[j] += xtl2.geometry[0][j+3*i]; }
    coord[0] /= (double)xtl2.natom[0];
    coord[1] /= (double)xtl2.natom[0];
    coord[2] /= (double)xtl2.natom[0];
    state[6] = coord[0];
    state[7] = coord[1];
    state[8] = coord[2];
    state[9] = 1.0;
    state[10] = 0.0;
    state[11] = 0.0;
    state[12] = 0.0;
    for(int i=0 ; i<xtl2.natom[0] ; i++)
    for(int j=0 ; j<3 ; j++)
    { xtl2.geometry[0][j+3*i] -= coord[j]; }

    // align & center the remaining molecules
    double *geo = (double*)malloc(sizeof(double)*xtl2.natom[0]*3);
    for(int i=1 ; i<xtl2.nmol ; i++)
    {
        // transform molecular coordinates to be consistent w/ lattice vectors
        for(int j=0 ; j<xtl2.natom[0] ; j++)
        {
            geo[0+j*3] = xtl->Xcord[j+i*xtl2.natom[0]];
            geo[1+j*3] = xtl->Ycord[j+i*xtl2.natom[0]];
            geo[2+j*3] = xtl->Zcord[j+i*xtl2.natom[0]];
        }
        if(family == 0)
        { dormqr_(&left, &trans, &three, xtl2.natom, &three, latvec2, &three, tau, geo, &three, work, &lwork, &info); }

        // extract center of molecule
        for(int j=0 ; j<3 ; j++)
        {
            state[j+6 + 7*i] = 0.0;
            for(int k=0 ; k<xtl2.natom[0] ; k++)
            { state[j+6 + 7*i] += geo[j+k*3]; }
            state[j+6 + 7*i] /= (double)xtl2.natom[0];
        }
        for(int j=0 ; j<xtl2.natom[0] ; j++)
        for(int k=0 ; k<3 ; k++)
        { geo[k + j*3] -= state[k+6 + 7*i]; }

        // form rotation matrix (Kabsch algorithm)
        char all = 'A';
        double sv[3], rot[9], U[9], VT[9], zero = 0.0, real_one = 1.0;
        dgemm_(&notrans, &trans, &three, &three, xtl2.natom, &real_one, geo, &three, xtl2.geometry[0], &three, &zero, rot, &three);
        dgesvd_(&all, &all, &three, &three, rot, &three, sv, U, &three, VT, &three, work, &lwork, &info);
        dgemm_(&notrans, &notrans, &three, &three, &three, &real_one, U, &three, VT, &three, &zero, rot, &three);

        // check for inversion (determinant test)
        double det = rot[0+0*3]*rot[1+1*3]*rot[2+2*3]
                    -rot[0+0*3]*rot[1+2*3]*rot[2+1*3]
                    +rot[0+1*3]*rot[1+2*3]*rot[2+0*3]
                    -rot[0+1*3]*rot[1+0*3]*rot[2+2*3]
                    +rot[0+2*3]*rot[1+0*3]*rot[2+1*3]
                    -rot[0+2*3]*rot[1+1*3]*rot[2+0*3];
        if(det < 0.0)
        {
            xtl2.invert[i] = -1;
            for(int j=0 ; j<9 ; j++)
            { rot[j] *= (double)xtl2.invert[i]; }
        }

        // extract quaternion vector
        matrix2quaternion(rot, state + 9 + 7*i);
    }

    // run the optimizer
    optimize(&xtl2, state, family);

    // convert the result back to the original format
    for(int i=0 ; i<9 ; i++)
    { latvec[i] = 0.0; }
    latvec[0 + 0*3] = state[0];
    latvec[0 + 1*3] = state[1];
    latvec[1 + 1*3] = state[2];
    latvec[0 + 2*3] = state[3];
    latvec[1 + 2*3] = state[4];
    latvec[2 + 2*3] = state[5];
    if(family == 0)
    { dormqr_(&left, &notrans, &three, &three, &three, latvec2, &three, tau, latvec, &three, work, &lwork, &info); }

    for(int i=0 ; i<3 ; i++)
    for(int j=0 ; j<3 ; j++)
    { xtl->lattice_vectors[i][j] = latvec[j + i*3]; }

    for(int i=0 ; i<xtl2.nmol ; i++)
    for(int j=0 ; j<xtl2.natom[0] ; j++)
    {
        double local[3];
        for(int k=0 ; k<3 ; k++)
        { local[k] = (double)xtl2.invert[i]*xtl2.geometry[0][k+3*j]; }
        position(local, state + 6 + 7*i, coord);
        if(family == 0)
        { dormqr_(&left, &notrans, &three, &one, &three, latvec2, &three, tau, coord, &three, work, &lwork, &info); }

        xtl->Xcord[j+i*xtl2.natom[0]] = coord[0];
        xtl->Ycord[j+i*xtl2.natom[0]] = coord[1];
        xtl->Zcord[j+i*xtl2.natom[0]] = coord[2];
    }
/*
printf("final geometry:\n");
for(int i=0 ; i<3 ; i++)
{
    printf("latt %d %f %f %f\n", i, xtl->lattice_vectors[i][0], xtl->lattice_vectors[i][1], xtl->lattice_vectors[i][2]);
}
for(int j=0 ; j<xtl->Z*xtl->num_atoms_in_molecule ; j++)
{
    printf("pos %d %f %f %f\n", j, xtl->Xcord[j], xtl->Ycord[j], xtl->Zcord[j]);
}
*/
    // de-allocate memory for the temporary crystal data structure & state vector
    free_molecular_crystal(&xtl2);
    free(state);
    free(geo);
    free(work);
}

// NOTE: rows/columns of the cutoff_matrix are over all atoms in the unit cell
void optimize_cocrystal(cocrystal *xtl, double *cutoff_matrix, int family)
{
/*
printf("initial geometry:\n");
for(int i=0 ; i<3 ; i++)
{
    printf("latt %d %f %f %f\n", i, xtl->lattice_vectors[i][0], xtl->lattice_vectors[i][1], xtl->lattice_vectors[i][2]);
}
for(int j=0 ; j<xtl->n_atoms ; j++)
{
    printf("pos %d %f %f %f\n", j, xtl->Xcord[j], xtl->Ycord[j], xtl->Zcord[j]);
}
*/
    // test of lattice vector formatting
    if(family != 0 && (xtl->lattice_vectors[0][1] != 0.0 || xtl->lattice_vectors[0][2] != 0.0 || xtl->lattice_vectors[1][2] != 0.0))
    { printf("ERROR: incorrect lattice vector format in optimize_crystal"); exit(1); }

    // allocate memory for the temporary crystal data structure & state vector
    struct molecular_crystal xtl2;
    xtl2.ntype = xtl->n_mol_types;
    xtl2.nmol = xtl->n_mols;
    xtl2.natom = (int*)malloc(sizeof(int)*xtl2.ntype);
    xtl2.geometry = (double**)malloc(sizeof(double*)*xtl2.ntype);
    xtl2.collide = (double***)malloc(sizeof(double**)*xtl2.ntype);
    for(int i=0 ; i<xtl2.ntype ; i++)
    { xtl2.collide[i] = (double**)malloc(sizeof(double*)*xtl2.ntype); }
    xtl2.type = (int*)malloc(sizeof(int)*xtl2.nmol);
    xtl2.invert = (int*)malloc(sizeof(int)*xtl2.nmol);
    for(int i=0 ; i<xtl2.nmol ; i++)
    { xtl2.type[i] = xtl->mol_types[i]; xtl2.invert[i] = 1; }
    double *state = (double*)malloc(sizeof(double)*(6 + 7*xtl2.nmol));
    int max_num = 0;
    for(int i=0 ; i<xtl2.nmol ; i++)
    { if(xtl->n_atoms_in_mol[i] > max_num) { max_num = xtl->n_atoms_in_mol[i]; } }

    // form rotation matrix to align lattice vectors (QR decomposition)
    double latvec[9], latvec2[9];
    for(int i=0 ; i<3 ; i++)
    for(int j=0 ; j<3 ; j++)
    { latvec[j + i*3] = xtl->lattice_vectors[i][j]; }
    for(int i=0 ; i<9 ; i++)
    { latvec2[i] = latvec[i]; }
    double tau[3];
    int lwork = max_num+100, info, three = 3, one = 1;
    double *work = (double*)malloc(sizeof(double)*lwork);
    if(family == 0)
    { dgeqrf_(&three, &three, latvec2, &three, tau, work, &lwork, &info); }
    state[0] = latvec2[0 + 0*3];
    state[1] = latvec2[0 + 1*3];
    state[2] = latvec2[1 + 1*3];
    state[3] = latvec2[0 + 2*3];
    state[4] = latvec2[1 + 2*3];
    state[5] = latvec2[2 + 2*3];

    // extract a reference geometry for every molecule type
    int imol, jmol;
    double coord[3];
    char trans = 'T', notrans = 'N', left = 'L';
    for(int i=0 ; i<xtl2.ntype ; i++)
    {
        // find the first molecule of the active type
        imol = jmol = 0;
        while(xtl2.type[imol] != i)
        {
            jmol += xtl->n_atoms_in_mol[imol];
            imol++;
            if(imol == xtl2.nmol)
            { printf("ERROR: molecule type not found in optimize_cocrystal"); exit(1); }
        }

        xtl2.natom[i] = xtl->n_atoms_in_mol[imol];
        xtl2.geometry[i] = (double*)malloc(sizeof(double)*3*xtl2.natom[i]);

        // store the reference molecule
        for(int j=0 ; j<xtl2.natom[i] ; j++)
        {
            xtl2.geometry[i][0+3*j] = xtl->Xcord[j+jmol];
            xtl2.geometry[i][1+3*j] = xtl->Ycord[j+jmol];
            xtl2.geometry[i][2+3*j] = xtl->Zcord[j+jmol];
        }
        if(family == 0)
        { dormqr_(&left, &trans, &three, xtl2.natom, &three, latvec2, &three, tau, xtl2.geometry[i], &three, work, &lwork, &info); }

        // extract geometric center & center reference geometry
        coord[0] = coord[1] = coord[2] = 0.0;
        for(int j=0 ; j<xtl2.natom[i] ; j++)
        for(int k=0 ; k<3 ; k++)
        { coord[k] += xtl2.geometry[i][k+3*j]; }
        coord[0] /= (double)xtl2.natom[i];
        coord[1] /= (double)xtl2.natom[i];
        coord[2] /= (double)xtl2.natom[i];
        for(int j=0 ; j<xtl2.natom[i] ; j++)
        for(int k=0 ; k<3 ; k++)
        { xtl2.geometry[i][k+3*j] -= coord[k]; }
    }

    // allocate memory to the internal cutoff matrices
    for(int i=0 ; i<xtl2.ntype ; i++)
    for(int j=0 ; j<xtl2.ntype ; j++)
    { xtl2.collide[i][j] = (double*)malloc(sizeof(double)*xtl2.natom[i]*xtl2.natom[j]); }

    // translate the cutoff matrix formats
    imol = 0;
    for(int i=0 ; i<xtl2.nmol ; i++)
    {
        jmol = 0;
        for(int j=0 ; j<xtl2.nmol ; j++)
        {
            for(int k=0 ; k<xtl2.natom[xtl2.type[j]] ; k++)
            for(int l=0 ; l<xtl2.natom[xtl2.type[i]] ; l++)
            {
                xtl2.collide[xtl2.type[i]][xtl2.type[j]][l + k*xtl2.natom[xtl2.type[i]]] = cutoff_matrix[l+imol + (k+jmol)*xtl->n_atoms];
            }
            jmol += xtl2.natom[j];
        }
        imol += xtl2.natom[i];
    }

    // align & center all molecules (redundant for reference molecules)
    double *geo = (double*)malloc(sizeof(double)*max_num*3);
    jmol = 0;
    for(int i=0 ; i<xtl2.nmol ; i++)
    {
        imol = xtl2.type[i];

        // transform molecular coordinates to be consistent w/ lattice vectors
        for(int j=0 ; j<xtl2.natom[imol] ; j++)
        {
            geo[0+j*3] = xtl->Xcord[j+jmol];
            geo[1+j*3] = xtl->Ycord[j+jmol];
            geo[2+j*3] = xtl->Zcord[j+jmol];
        }
        if(family == 0)
        { dormqr_(&left, &trans, &three, xtl2.natom, &three, latvec2, &three, tau, geo, &three, work, &lwork, &info); }

        // extract center of molecule
        for(int j=0 ; j<3 ; j++)
        {
            state[j+6 + 7*i] = 0.0;
            for(int k=0 ; k<xtl2.natom[imol] ; k++)
            { state[j+6 + 7*i] += geo[j+k*3]; }
            state[j+6 + 7*i] /= (double)xtl2.natom[imol];
        }
        for(int j=0 ; j<xtl2.natom[imol] ; j++)
        for(int k=0 ; k<3 ; k++)
        { geo[k + j*3] -= state[k+6 + 7*i]; }

        // form rotation matrix (Kabsch algorithm)
        char all = 'A';
        double sv[3], rot[9], U[9], VT[9], zero = 0.0, real_one = 1.0;
        dgemm_(&notrans, &trans, &three, &three, xtl2.natom, &real_one, geo, &three, xtl2.geometry[imol], &three, &zero, rot, &three);
        dgesvd_(&all, &all, &three, &three, rot, &three, sv, U, &three, VT, &three, work, &lwork, &info);
        dgemm_(&notrans, &notrans, &three, &three, &three, &real_one, U, &three, VT, &three, &zero, rot, &three);

        // check for inversion (determinant test)
        double det = rot[0+0*3]*rot[1+1*3]*rot[2+2*3]
                    -rot[0+0*3]*rot[1+2*3]*rot[2+1*3]
                    +rot[0+1*3]*rot[1+2*3]*rot[2+0*3]
                    -rot[0+1*3]*rot[1+0*3]*rot[2+2*3]
                    +rot[0+2*3]*rot[1+0*3]*rot[2+1*3]
                    -rot[0+2*3]*rot[1+1*3]*rot[2+0*3];
        if(det < 0.0)
        {
            xtl2.invert[i] = -1;
            for(int j=0 ; j<9 ; j++)
            { rot[j] *= (double)xtl2.invert[i]; }
        }

        // extract quaternion vector
        matrix2quaternion(rot, state + 9 + 7*i);

        // update global atom index
        jmol += xtl2.natom[imol];
    }

    // run the optimizer
    optimize(&xtl2, state, family);

    // convert the result back to the original format
    for(int i=0 ; i<9 ; i++)
    { latvec[i] = 0.0; }
    latvec[0 + 0*3] = state[0];
    latvec[0 + 1*3] = state[1];
    latvec[1 + 1*3] = state[2];
    latvec[0 + 2*3] = state[3];
    latvec[1 + 2*3] = state[4];
    latvec[2 + 2*3] = state[5];
    if(family == 0)
    { dormqr_(&left, &notrans, &three, &three, &three, latvec2, &three, tau, latvec, &three, work, &lwork, &info); }

    for(int i=0 ; i<3 ; i++)
    for(int j=0 ; j<3 ; j++)
    { xtl->lattice_vectors[i][j] = latvec[j + i*3]; }

    jmol = 0;
    for(int i=0 ; i<xtl2.nmol ; i++)
    {
        imol = xtl2.type[i];
        for(int j=0 ; j<xtl2.natom[imol] ; j++)
        {
            double local[3];
            for(int k=0 ; k<3 ; k++)
            { local[k] = (double)xtl2.invert[i]*xtl2.geometry[imol][k+3*j]; }
            position(local, state + 6 + 7*i, coord);
            if(family == 0)
            { dormqr_(&left, &notrans, &three, &one, &three, latvec2, &three, tau, coord, &three, work, &lwork, &info); }

            xtl->Xcord[j+jmol] = coord[0];
            xtl->Ycord[j+jmol] = coord[1];
            xtl->Zcord[j+jmol] = coord[2];
        }
        jmol += xtl2.natom[imol];
    }
/*
printf("final geometry:\n");
for(int i=0 ; i<3 ; i++)
{
    printf("latt %d %f %f %f\n", i, xtl->lattice_vectors[i][0], xtl->lattice_vectors[i][1], xtl->lattice_vectors[i][2]);
}
for(int j=0 ; j<xtl->n_atoms ; j++)
{
    printf("pos %d %f %f %f\n", j, xtl->Xcord[j], xtl->Ycord[j], xtl->Zcord[j]);
}
*/
    // de-allocate memory for the temporary crystal data structure & state vector
    free_molecular_crystal(&xtl2);
    free(state);
    free(geo);
    free(work);
}
/*
// test main
int main(void)
{
    // construct test crystal
    struct molecular_crystal xtl;
    xtl.ntype = 2;
    xtl.nmol = 3;
    xtl.natom = (int*)malloc(sizeof(int)*2);
    xtl.natom[0] = 2;
    xtl.natom[1] = 3;
    xtl.geometry = (double**)malloc(sizeof(double*)*2);
    for(int i=0 ; i<2 ; i++)
    {
        xtl.geometry[i] = (double*)malloc(sizeof(double)*3*xtl.natom[i]);
    }
    xtl.geometry[0][0 + 3*0] = 0.0;
    xtl.geometry[0][1 + 3*0] = 0.0;
    xtl.geometry[0][2 + 3*0] = 0.0;
    xtl.geometry[0][0 + 3*1] = 2.0;
    xtl.geometry[0][1 + 3*1] = 0.0;
    xtl.geometry[0][2 + 3*1] = 0.0;
    xtl.geometry[1][0 + 3*0] = 0.0;
    xtl.geometry[1][1 + 3*0] = 0.0;
    xtl.geometry[1][2 + 3*0] = 0.0;
    xtl.geometry[1][0 + 3*1] = 0.0;
    xtl.geometry[1][1 + 3*1] = 2.0;
    xtl.geometry[1][2 + 3*1] = 0.0;
    xtl.geometry[1][0 + 3*2] = 0.0;
    xtl.geometry[1][1 + 3*2] = 0.0;
    xtl.geometry[1][2 + 3*2] = 2.0;

    xtl.collide = (double***)malloc(sizeof(double**)*2);
    for(int i=0 ; i<2 ; i++)
    {
        xtl.collide[i] = (double**)malloc(sizeof(double*)*2);

        for(int j=0 ; j<2 ; j++)
        {
            xtl.collide[i][j] = (double*)malloc(sizeof(double)*xtl.natom[i]*xtl.natom[j]);
            for(int k=0 ; k<xtl.natom[i]*xtl.natom[j] ; k++)
            { xtl.collide[i][j][k] = 3.0+0.1*(double)(i+j+k); }
        }
    }
    xtl.type = (int*)malloc(sizeof(int)*3);
    xtl.type[0] = 0;
    xtl.type[1] = 0;
    xtl.type[2] = 1;

    xtl.invert = (int*)malloc(sizeof(int)*3);
    xtl.invert[0] = 1;
    xtl.invert[1] = -1;
    xtl.invert[2] = 1;

    // construct test state
    double state[6+7*3];
    state[0] = 10.0;
    state[1] = 1.0;
    state[2] = 10.0;
    state[3] = -1.0;
    state[4] = 0.5;
    state[5] = 10.0;

    state[6+0+7*0] = 0.0;
    state[6+1+7*0] = 0.1;
    state[6+2+7*0] = 0.0;
    state[6+3+7*0] = 1.0;
    state[6+4+7*0] = 0.2;
    state[6+5+7*0] = 0.1;
    state[6+6+7*0] = 0.3;

    state[6+0+7*1] = 0.2;
    state[6+1+7*1] = 5.0;
    state[6+2+7*1] = 0.5;
    state[6+3+7*1] = 0.05;
    state[6+4+7*1] = 0.9;
    state[6+5+7*1] = 0.1;
    state[6+6+7*1] = 0.05;

    state[6+0+7*2] = 1.0;
    state[6+1+7*2] = 2.0;
    state[6+2+7*2] = 5.0;
    state[6+3+7*2] = 0.0;
    state[6+4+7*2] = -0.3;
    state[6+5+7*2] = 1.0;
    state[6+6+7*2] = 0.2;

// pair energy derivative test

double grad1[7], grad2[7], hess11[49], hess22[49], hess12[49];
double grad1_test[7], grad2_test[7], hess11_test[49], hess22_test[49], hess12_test[49];
pair_energy_derivative(xtl.natom[0],xtl.natom[1],xtl.invert[0],xtl.invert[1],xtl.geometry[0], xtl.geometry[1],xtl.collide[0][1],state+6,state+13,
                       grad1, grad2, hess11, hess22, hess12);
pair_energy_derivative_test(xtl.natom[0],xtl.natom[1],xtl.invert[0],xtl.invert[1],xtl.geometry[0], xtl.geometry[1],xtl.collide[0][1],state+6,state+13,
                            grad1_test, grad2_test, hess11_test, hess22_test, hess12_test);
printf("grad tests:\n");
for(int i=0 ; i<7 ; i++)
{
    printf("%d %e %e | %e %e\n", i, grad1[i], grad1_test[i], grad2[i], grad2_test[i]);
}
for(int i=0 ; i<49 ; i++)
{
    printf("%d %e %e | %e %e | %e %e\n", i, hess11[i], hess11_test[i], hess22[i], hess22_test[i], hess12[i], hess12_test[i]);
}
exit(1);

    double energy = total_energy(&xtl, state);
    printf("energy = %e\n",energy);

    double grad0[6+7*3], grad[6+7*3], hess0[(6+7*3)*(6+7*3)], hess[(6+7*3)*(6+7*3)];
    total_energy_derivative_test(&xtl, state, grad0, hess0);
    total_energy_derivative(&xtl, state, grad, hess);
    for(int i=0 ; i<6+7*3 ; i++)
    {
        if(fabs(grad0[i] - grad[i]) > 1e-4*fabs(grad0[i] + grad[i]))
        { printf("grad[%d] = %e %e\n", i, grad0[i], grad[i]); }
    }
    for(int i=0 ; i<(6+7*3)*(6+7*3) ; i++)
    {
        if(fabs(hess0[i] - hess[i]) > 1e-4*fabs(hess0[i] + hess[i]))
        { printf("hess[%d] = %e %e\n", i, hess0[i], hess[i]); }
    }

    optimize(&xtl, state);

    free_molecular_crystal(&xtl);
    return 0;
}
*/