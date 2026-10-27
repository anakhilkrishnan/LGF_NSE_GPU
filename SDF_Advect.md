# Signed Distance Function (SDF) implementation with WENO5 advection in AMReX

Akhil Krishnan\
October 27, 2026\
Indian Institute of Science

## Code Structure

The code is split into 4 files:

1. `interfaceTracker.cpp` : contains routines for initialization, filling boundary conditions, computing RHS for time integrator, updating with the snug domain, and checking if the unclipped band is inside the snug domain.
2. `InterfaceTracker.H` : contains class definition and all dependencies.
3. `InterfaceDiscretization.H` : contains WENO5/HOUC5 discretization functions.
4. `InitialSDF.H` : contains functions that provide expressions for initial conditions, including primitive shapes and combinations and the 'active case'.

The time integration is in `ProjectionWorkspace` class, `main.cpp` handles initialization and `IOManager` class handles plotting.

## Implementation notes

### Inputs to the solver

1. Initial condition for $\psi$ (set in `InitialSDF.H` file at compile time).
2. `bool` switch to disable interface tracking if not needed.
3. `scheme` (WENO5 or HOUC5) as a runtime variable.
4. $\psi_\infty$ value.

### Data handling and boundary conditions

1. Signed Distance Function ($\psi$) stored in cell-centered index type, in every cell in the snug domain, including the buffer.
2. Each field is stored as a component of the same `amrex::MultiFab` object, with 3 ghost cells.
3. $\psi > 0$ for inside the shape, $\psi = 0$ for interface, and $\psi < 0$ for outside.
4. $\psi$ is clipped between limits ($\pm \psi_{\infty}$) at initialization.
5. The $\psi$ band ($|\psi| < \psi_\infty$) is required to stay inside the snug domain; the run aborts at initialization, restart or regrid otherwise. It should also stay inside the flow’s support region, since velocity in the buffer is inaccurate, but this is not checked. Ghost cells that overlap a neighbouring box take its values; ghost cells outside the snug domain are filled by linear extrapolation from the nearest valid cell of the same box.
6. Each field is plotted for post processing.

### Advection scheme

1. WENO5/HOUC5 implemented with the same scheme as `Own-LS-Option-3D-Test-Advection.h`. Algebraically regrouped as 3 candidates of third-order stencils, computed from face-gradients of $\psi$.
2. The velocities are averaged to cell-centers.
3. Only the upwind derivative is computed. (`Own-LS-Option-3D-Test-Advection.h` computes both `Psix_L` and `Psix_R` for each cell) 
4. HOUC5 is computed identically to WENO5 with the optimal weights (0.1, 0.6, 0.3) without any smoothness indicators.
5. At each stage, velocity ($\vec{u}$) obtained (as a sum of velocity perturbation from the projection method and freestream) from IF-HERK integration is used to advect $\psi$ using the same Butcher Tableau as the HERK, without the action of the integrating factor.
6. On every snug-domain update, the regrid routine copies $\psi$ wherever the old and new grids overlap and assigns $-\psi_\infty$ to new cells. Beforehand it checks that removed cells, and cells on the edge of the old domain, all have $\psi \leq -\psi_\infty + 3\Delta x $ (the stencil reach), and aborts otherwise.

### Reinitialization

Yet to be implemented.

### Tube/Band masking

Yet to be implemented.

### Refine at restart

Yet to be implemented for the signed distance function field, currently aborts run if attempted.
