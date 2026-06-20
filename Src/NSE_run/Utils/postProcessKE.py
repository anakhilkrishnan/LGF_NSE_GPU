import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys


# User variables
filepath = "kinetic_energy.dat"

# Declare names for components
comp_names = ["x-component", "y-component", "z-component"]

# Read the tab-separated data
try:
    df = pd.read_csv(filepath, sep='\t')
except FileNotFoundError:
    print(f"Error: Could not find '{filepath}'.")
    sys.exit(1)

# Clean up column names (strip any accidental whitespace)
df.columns = df.columns.str.strip()

# Detect the number of spatial dimensions tracked in the file
comp_cols = [col for col in df.columns if 'totalKE_dir_comp' in col]
n_dim = len(comp_cols)
print(f"Successfully loaded data. Detected {n_dim} spatial dimensions.")

# Create a multi-panel plot
# Fig 1: Component-wise KE
# Fig 2: Component-wise relative error
fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

plt.suptitle("KE data using separate KE discretization for evolution")

# # --- Plot 1: Component-wise Energy Breakdown ---
ax = axes[0]
colors = ['orange', 'green', 'purple']
for idim in range(n_dim):
    dir_col = f'totalKE_dir_comp{idim}'
    evol_col = f'totalKE_evol_comp{idim}'
    
    if dir_col in df.columns:
        ax.plot(df['Step'], df[dir_col], color=colors[idim % 3], 
                label=f'{comp_names[idim]} Direct')
    if evol_col in df.columns:
        ax.plot(df['Step'], df[evol_col], color=colors[idim % 3], linestyle='--', alpha=0.7, 
                label=f'{comp_names[idim]} Evolved')


ax.set_ylabel('Component KE')
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend()

# # --- Plot 2: Total Kinetic Energy (Direct vs Evolved) ---
# Computing relative error
ax = axes[1]
for idim in range(n_dim):
    dir_col = f'totalKE_dir_comp{idim}'
    evol_col = f'totalKE_evol_comp{idim}'
    
    if dir_col in df.columns:

        tKEcomp_relerr = (df[dir_col] - df[evol_col])/(df[dir_col])
        ax.plot(df['Step'], tKEcomp_relerr, color=colors[idim % 3],
                label=f'{comp_names[idim]} Relative Error')
        
ax.set_xlabel('Timestep')
ax.set_ylabel('Relative Error in KE')
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend()

plt.tight_layout()
plt.savefig('kinetic_energy_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
