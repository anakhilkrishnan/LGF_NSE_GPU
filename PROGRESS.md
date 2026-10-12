# Developer progress report

Akhil Krishnan\
October 12, 2026\
Indian Institute of Science

## Ongoing work

A *Sharp Interface-Capturing method* is the next immediate goal. Involves:

1. A scalar field like a *signed distance function* (SDF) which is advected with the flow.
2. A high-order discretization of the convection of the scalar field, such as WENO 5 or HOUC (*High-Order Upstream Convective*) schemes.

## Yet to implement

1. `n_cell` and `max_grid_size` need to take `IntVect()` input.
2. `n_comp` and `n_host` need to be removed from inputs, must be deliberately set in each required location (`n_ghost` needs to be set based on `n_IF`).
3. CFL must be checked in each step.
4. `rk_stages` in inputs is not needed.
5. Validate the `precomputeIFs()` function.
6. Set all diagnostic norms correctly `divU_max_end`, `vel_refresh_err_max_norm` and so on.
7. Compute `q_max` correctly, no longer in inputs.
8. `n_IF` must be removed from inputs, in favour of `eps_IF` as the tolerance for the IF tail.
9. Report kinetic energy.
10. Move the IF-HERK code out of `ProjectionWorkspace` into a separate `IFHERK` class, similar to `IF_HERK_Valdiate` repository. 
11. Plots represent perturbation velocity and total pressure. Switch to full velocity and static pressure plots.
12. Report `divInitVort` to ensure that flux-averaged initialization works.
13. Scheme C from Liska and Colonius' paper doesn't work. Check why and remove conflicting assert that is unused.

## Yet to decide

1. A Fast-Multipole Method library would make a great pairing with the solver, especially in 3D. Patch-wise FFT also almost as good, keeping code dependent only on AMReX in 2D and 3D. 
