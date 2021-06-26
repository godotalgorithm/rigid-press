# rigid-press

This is a simple structure optimization program for molecular crystals that freezes the internal geometry of molecules
but optimizes their position and orientation along with the crystal lattice vectors to minimize crystal volume.
A hard-sphere interaction between molecules defined by interatomic cutoff distances is regularized into a smooth potential
to enable conventional optimization techniques for a differentiable objective function.

This implementation includes interfaces for use with [Genarris](https://github.com/ritwit/cgenarris),
and it is expected that it will be incorporated into future versions of Genarris.
The first stage of structure generation in Genarris is based on rejection sampling of randomly generated crystals,
which can result in very low acceptance rates for certain problems.
The intent of this program is to allow for pre-packing of randomly generated crystals to increase the acceptance rate.
