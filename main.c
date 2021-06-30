// test example for the rigid-body optimizer

#include <stdio.h>
#include <stdlib.h>

#include "crystal.h"
#include "cocrystal.h"
#include "rigid-press.h"

void read_vector(int size, char *path, double *vec)
{
    FILE *file_ptr = fopen(path, "r");

    for(int i=0 ; i<size ; i++)
    { fscanf(file_ptr, "%lf", vec+i); }

    fclose(file_ptr);
}

void read_crystal(char *path, crystal *xtl)
{
    char line[1000], dummy[1000];
    FILE *file_ptr = fopen(path, "r");

    int i=0, j=0;
    while(!feof(file_ptr))
    {
        char *out = fgets(line, 1000, file_ptr);
        if(out == NULL)
        { break; }

        switch(line[0])
        {
            case 'l':
            sscanf(line, "%s %f %f %f", dummy, xtl->lattice_vectors[j], xtl->lattice_vectors[j]+1, xtl->lattice_vectors[j]+2);
            j++;
            break;

            case 'a':
            sscanf(line, "%s %f %f %f %s", dummy, xtl->Xcord+i, xtl->Ycord+i, xtl->Zcord+i, xtl->atoms+2*i);
            i++;
            break;

            case '#':
            break;

            default:
            printf("ERROR: parsing line (%s)\n", line);
            exit(1);
        }
    }
    fclose(file_ptr);
}

int main(void)
{
    // read in example 1
    crystal xtl;
    xtl.num_atoms_in_molecule = 12;
    xtl.Z = 2;
    int num_atoms = xtl.Z*xtl.num_atoms_in_molecule;
    xtl.Xcord = (float*)malloc(sizeof(float)*num_atoms);
    xtl.Ycord = (float*)malloc(sizeof(float)*num_atoms);
    xtl.Zcord = (float*)malloc(sizeof(float)*num_atoms);
    xtl.atoms = (char*)malloc(sizeof(char)*(2*num_atoms+1));
    double *cutmat = (double*)malloc(sizeof(double)*num_atoms*num_atoms);
    for(int i=0 ; i<=2*num_atoms ; i++)
    { xtl.atoms[i] = ' '; }
    read_crystal("sample_structures/Example1/geometry.in", &xtl);
    read_vector(num_atoms*num_atoms, "sample_structures/Example1/cutoff_matrix.txt", cutmat);
    optimize_crystal(&xtl, cutmat);
    free(xtl.Xcord);
    free(xtl.Ycord);
    free(xtl.Zcord);
    free(xtl.atoms);
    free(cutmat);

    // read in example 2
    xtl.num_atoms_in_molecule = 12;
    xtl.Z = 4;
    num_atoms = xtl.Z*xtl.num_atoms_in_molecule;
    xtl.Xcord = (float*)malloc(sizeof(float)*num_atoms);
    xtl.Ycord = (float*)malloc(sizeof(float)*num_atoms);
    xtl.Zcord = (float*)malloc(sizeof(float)*num_atoms);
    xtl.atoms = (char*)malloc(sizeof(char)*(2*num_atoms+1));
    cutmat = (double*)malloc(sizeof(double)*num_atoms*num_atoms);
    for(int i=0 ; i<=2*num_atoms ; i++)
    { xtl.atoms[i] = ' '; }
    read_crystal("sample_structures/Example2/geometry.in", &xtl);
    read_vector(num_atoms*num_atoms, "sample_structures/Example2/cutoff_matrix.txt", cutmat);
    optimize_crystal(&xtl, cutmat);
    free(xtl.atoms);

    // convert example 2 to cocrystal format
    cocrystal coxtl;
    coxtl.n_mol_types = 2;
    coxtl.n_mols = 4;
    coxtl.n_atoms = 48;
    coxtl.Xcord = xtl.Xcord;
    coxtl.Ycord = xtl.Ycord;
    coxtl.Zcord = xtl.Zcord;
    coxtl.mol_types = (int*)malloc(sizeof(int)*4);
    coxtl.mol_types[0] = 0;
    coxtl.mol_types[1] = 1;
    coxtl.mol_types[2] = 0;
    coxtl.mol_types[3] = 1;
    coxtl.n_atoms_in_mol = (int*)malloc(sizeof(int)*4);
    for(int i=0 ; i<4 ; i++)
    { coxtl.n_atoms_in_mol[i] = 12; }
    for(int i=0 ; i<3 ; i++)
    for(int j=0 ; j<3 ; j++)
    { coxtl.lattice_vectors[i][j] = xtl.lattice_vectors[i][j]; }
    optimize_cocrystal(&coxtl, cutmat);
    free(coxtl.mol_types);
    free(coxtl.n_atoms_in_mol);
    free(xtl.Xcord);
    free(xtl.Ycord);
    free(xtl.Zcord);
    free(cutmat);

    return 0;
}
