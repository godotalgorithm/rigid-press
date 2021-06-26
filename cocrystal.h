#ifndef _COCRYSTAL_H_
#define _COCRYSTAL_H_

typedef struct
{
    float lattice_vectors[3][3];
    float *Xcord;                  // Size n_atoms
    float *Ycord;                  // Size n_atoms
    float *Zcord;                  // Size n_atoms
    float (*com)[3];               // Size n_mols
    char  *atoms;                  // Size 2 x n_atoms
    int   *mol_index;              // Size n_mols
    int   *mol_types;              // Size n_mols
    int   *n_atoms_in_mol;         // Size n_mol_types
    int   *wyckoff_position;       // Size n_mol_types
    int   *stoic;                  // Size n_mol_types
    int   n_mols;                  // Number of molecules
    int   n_mol_types;             // Number of molecule types
    int   n_atoms;                 // Total number of atoms
    int   spg;
    int   Z;
    int   Zp;
}cocrystal;

#endif  // Cocrystal.h
