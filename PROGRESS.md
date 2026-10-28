# Developer progress report

Akhil Krishnan\
October 12, 2026\
Indian Institute of Science

## Ongoing work

A *Sharp Interface-Capturing method* is the next immediate goal. Involves:

1. A scalar field like a *signed distance function* (SDF) which is advected with the flow.
2. A high-order discretization of the convection of the scalar field, such as WENO 5 or HOUC (*High-Order Upstream Central*) schemes.

## Yet to implement

1. ~~`n_cell` and `max_grid_size` need to take `amrex::IntVect` input, and `dom_lo` and `dom_hi` take `amrex::RealVect` input~~.
2. `n_comp` and `n_ghost` need to be removed from inputs, must be deliberately set in each required location (`n_ghost` needs to be set based on `n_IF`).
3. CFL must be checked in each step.
4. ~~`rk_stages` in inputs is not needed~~.
5. Validate the `precomputeIFs()` function.
6. Set all diagnostic norms correctly `divU_max_end`, `vel_refresh_err_max_norm` and so on.
7. Compute `q_max` correctly, no longer in inputs.
8. `n_IF` must be removed from inputs, in favour of `eps_IF` as the tolerance for the IF tail.
9. Report kinetic energy.
10. Move the IF-HERK code out of `ProjectionWorkspace` into a separate `IFHERK` class, similar to `IF_HERK_Valdiate` repository. 
11. Plots represent perturbation velocity and total pressure. Switch to full velocity and static pressure plots.
12. Report `divInitVort` to ensure that flux-averaged initialization works.
13. Scheme C from Liska and Colonius' paper doesn't work. Check why and remove conflicting assert that is unused.
14. Input parameters need to be split into different ParamParse streams (for ease of reading the inputs file).
15. Fused kernel launches and reduced memory traffic optimizations pending (see Claude thread on optimization).
16. ~~Allow for refinement at each restart (current idea: refine velocity using a tensor product cubic interpolation, perform vor2vel)~~.
17. ~~During search tagging, check if any tagged boxes are on the edge of the search domain. If so, abort~~. 
18. Add component-wise vorticity to plotfiles (!! Decide if necessary first !!).
19. Consider switching IO_int type variables to physical time instead of step count.
20. Fix the GPU job script for PARAM Rudra (check ThreeVortexRings Claude thread).
21. Make the CPU version OpenMP compatible (idea is to have 2 ranks per node, 1 per socket, similar to GPU setup).
22. Split the monolithic writeDiagnostics file into more meaningful quanities which are generalizable and meaningful on one side, and problem specific on the other.

## Yet to decide

1. A Fast-Multipole Method library would make a great pairing with the solver, especially in 3D. Patch-wise FFT also almost as good, keeping code dependent only on AMReX in 2D and 3D. 
2. In many cases in practice, the velocity refresh happens as often as every time step. The main reason for this is how the solver rebuilds D_xsoln every time D_supp changes. Decoupling this behaviour in some way will be novel to what Liska and Colonius suggested. 
3. For the refine at restart, look into interpolation strategies that preserve desired flow properties like mass/volume, momentum, energy etc.
