Process of using set of files for 3D kernel verification:
1. Table generation:
    python3 lgf3d_table_gen.py --n-lookup 16 --out lgf3d_table.inc --npy G_R16.npy
2. Asymptotic expansion generation:
    python3 lgf3d_asymptotic_gen.py --q-max 10 --out farField3D_10terms.inc
3. Sweep test for r_eps:
    python3 lgf3d_reps_sweep.py --inc farField3D_10terms.inc --npy G_R16.npy