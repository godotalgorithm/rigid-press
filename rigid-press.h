// Interface for the fast, approximate molecular crystal optimizer

#ifndef RIGID_PRESS_H
#define RIGID_PRESS_H

#include "crystal.h"

// optimizes a molecular crystal using a regularized rigid-body interaction
void optimize_crystal(crystal *xtl, // a molecular crystal in the Genarris crystal format [1]
                      double *cutoff_matrix); // distance cutoff between pairs of atoms in the crystallized molecule [xtl->num_atoms_in_molecule^2]
void optimize_cocrystal(cocrystal *xtl, // a molecular crystal in the Genarris co-crystal format [1]
                        double *cutoff_matrix); // distance cutoff between pairs of atoms in the crystallized molecule [complicated sizing...]

#endif
