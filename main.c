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


void read_matrix(int dim, char *path,  double *mat)
{
   FILE *file_ptr = fopen(path, "r");

   for(int i=0; i<dim ; i++)
   {
     for(int j=0 ; j<dim ; j++)
     { fscanf(file_ptr, "%lf", mat+12*i+j); }

     while(fgetc(file_ptr) != '\n'){}
   }

}


void print_matrix(int dim, double *mat)
{
  for(int i=0 ; i<dim ; i++)
  {
      for(int j=0 ; j<dim ; j++)
      { printf("%lf ", *(mat+12*i+j)); }

      printf("\n");
  }
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
    xtl.Xcord = (float*)malloc(sizeof(float)*xtl.Z*xtl.num_atoms_in_molecule);
    xtl.Ycord = (float*)malloc(sizeof(float)*xtl.Z*xtl.num_atoms_in_molecule);
    xtl.Zcord = (float*)malloc(sizeof(float)*xtl.Z*xtl.num_atoms_in_molecule);
    xtl.atoms = (char*)malloc(sizeof(char)*(2*xtl.Z*xtl.num_atoms_in_molecule+1));
    for(int i=0 ; i<=2*xtl.Z*xtl.num_atoms_in_molecule ; i++)
    { xtl.atoms[i] = ' '; }
    double cutmat[12*12];
    read_crystal("sample_structures/Example1/geometry.in", &xtl);
    //read_vector(12*12, "sample_structures/Example1/cutoff_matrix.txt", cutmat);
    read_matrix(12, "sample_structures/Example1/cutoff_matrix.txt", cutmat);
    print_matrix(12, cutmat);

    optimize_crystal(&xtl, cutmat);
    free(xtl.Xcord);
    free(xtl.Ycord);
    free(xtl.Zcord);
    free(xtl.atoms);

    exit(0);

    // read in example 2
    xtl.num_atoms_in_molecule = 12;
    xtl.Z = 4;
    xtl.Xcord = (float*)malloc(sizeof(float)*xtl.Z*xtl.num_atoms_in_molecule);
    xtl.Ycord = (float*)malloc(sizeof(float)*xtl.Z*xtl.num_atoms_in_molecule);
    xtl.Zcord = (float*)malloc(sizeof(float)*xtl.Z*xtl.num_atoms_in_molecule);
    xtl.atoms = (char*)malloc(sizeof(char)*(2*xtl.Z*xtl.num_atoms_in_molecule+1));
    for(int i=0 ; i<=2*xtl.Z*xtl.num_atoms_in_molecule ; i++)
    { xtl.atoms[i] = ' '; }
    read_crystal("sample_structures/Example2/geometry.in", &xtl);
    read_vector(12*12, "sample_structures/Example2/cutoff_matrix.txt", cutmat);
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
    double cutmat2[12*12*4];
    for(int i=0 ; i<12 ; i++)
    for(int j=0 ; j<12 ; j++)
    {
        cutmat2[i + j*24] = cutmat[i + j*12];
        cutmat2[12+i + j*24] = cutmat[i + j*12];
        cutmat2[i + (12+j)*24] = cutmat[i + j*12];
        cutmat2[12+i + (12+j)*24] = cutmat[i + j*12];
    }
    optimize_cocrystal(&coxtl, cutmat2);
    free(coxtl.mol_types);
    free(coxtl.n_atoms_in_mol);
    free(xtl.Xcord);
    free(xtl.Ycord);
    free(xtl.Zcord);

    return 0;
}
