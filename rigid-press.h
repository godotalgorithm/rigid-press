// Interface for the fast, approximate molecular crystal optimizer

#ifndef RIGID_PRESS_H
#define RIGID_PRESS_H

#include "crystal.h"

// optimizes a molecular crystal using a regularized rigid-body interaction
void optimize_crystal(crystal *xtl, // a molecular crystal in the Genarris structure format [1]
                      double *cutoff_matrix); // distance cutoff between pairs of atoms in the crystallized molecule [xtl->num_atoms_in_molecule^2]

#endif
