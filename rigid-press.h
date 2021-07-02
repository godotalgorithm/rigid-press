// Interface for the fast, approximate molecular crystal optimizer

#ifndef RIGID_PRESS_H
#define RIGID_PRESS_H

#include "crystal.h"
#include "cocrystal.h"

// crystal families & their lattice vector constraints
#define TRICLINIC    0 // no constraints on lattice vectors
#define MONOCLINIC   1 // (a, 0, 0), (0, b, 0), (c, 0, d)
#define ORTHORHOMBIC 2 // (a, 0, 0), (0, b, 0), (0, 0, c)
#define TETRAGONAL   3 // (a, 0, 0), (0, a, 0), (0, 0, b)
#define HEXAGONAL    4 // (a, 0, 0), (-a/2, a*sqrt(3)/2, 0), (0, 0, b)
#define CUBIC        5 // (a, 0, 0), (0, a, 0), (0, 0, a)

// optimizes a molecular crystal using a regularized rigid-body interaction
void optimize_crystal(crystal *xtl, // a molecular crystal in the Genarris crystal format [1]
                      double *cutoff_matrix, // distance cutoff between pairs of atoms in the crystallized molecule [(xtl->Z*xtl->num_atoms_in_molecule)^2]
                      int family); // crystal family (see above key)
void optimize_cocrystal(cocrystal *xtl, // a molecular crystal in the Genarris co-crystal format [1]
                        double *cutoff_matrix, // distance cutoff between pairs of atoms in the crystallized molecule [xtl->n_atoms^2]
                        int family); // crystal family (see above key)

#endif
