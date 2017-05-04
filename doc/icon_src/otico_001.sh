#!/usr/bin/env bash
#
#PBS -l nodes=1:ppn=32
#PBS -l walltime=48:00:00
#PBS -j oe
#
# Run Notes: ot test with vlct
#

cd "${PBS_O_WORKDIR:-./}"

nprocs=1
n_mhd_procs=1
npx=1
npy=1
npz=1

MPIRUN="${MPIRUN:-mpirun}"
MPI_OPTS="-n ${nprocs} ${MPI_OPTS}"
# GDB="${GDB:-gdb}"
LLDB="${LLDB:-lldb}"

export run_name="otico_001"
# export bin="./mhd_ot"
export bin="$HOME/dev/stage/libmrc-build/mhd/tests/mhd_ot"

cmd="${MPIRUN} ${MPI_OPTS} ${bin}"
# cmd="${MPIRUN} ${MPI_OPTS} xterm -e ${GDB} --args ${bin}"
# cmd="${MPIRUN} ${MPI_OPTS} xterm -e ${LLDB} ${bin} -- "

${cmd}                                                                       \
  --cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc \
  --ccccccccccccccccccccc  "GENERAL"                                         \
  --run                        ${run_name}                                   \
  --cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc \
  --ccccccccccccccccccccc  "GRID"                                            \
  --mrc_domain_npx ${npx} --mrc_domain_npy ${npy} --mrc_domain_npz ${npz}    \
  --mrc_domain_mx  256     --mrc_domain_my 256    --mrc_domain_mz  2         \
  --mrc_crds_lx    0.0    --mrc_crds_ly    0.0    --mrc_crds_lz    0.0       \
  --mrc_crds_hx    1.0    --mrc_crds_hy    1.0    --mrc_crds_hz    0.1       \
  --ggcm_mhd_crds_type         c                                             \
  --ggcm_mhd_crds_gen_type     mrc                                           \
  --cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc \
  --ccccccccccccccccccccc  "INITIAL CONDITION"                               \
  --ggcm_mhd_ic_type           ot                                            \
  --cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc \
  --ccccccccccccccccccccc  "NUMERICS"                                        \
  --ggcm_mhd_step_type         c3_double                                     \
  --ggcm_mhd_do_vgrupd         0                                             \
  --enforce_rrmin              0                                             \
  --cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc \
  --ccccccccccccccccccccc  "TIMESTEP"                                        \
  --dtmin                      5e-5                                          \
  --ggcm_mhd_step_do_nwst      1                                             \
  --ggcm_mhd_thx               0.4                                           \
  --mrc_ts_max_time            0.4                                           \
  --cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc \
  --ccccccccccccccccccccc  "SIM PARAMETERS"                                  \
  --ggcm_mhd_d_i               0.0                                           \
  --magdiffu                   const                                         \
  --diffconstant               0.0                                           \
  --cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc \
  --ccccccccccccccccccccc  "OUTPUT"                                          \
  --ggcm_mhd_step_debug_dump   0                                             \
  --mrc_ts_output_every_time   0.01                                          \
  --mrc_io_type                xdmf2                                         \
  --mrc_io_sw                  0                                             \
  --ggcm_mhd_diag_fields       rr:pp:v:b:j:e_cc:divb                         \
  --cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc \
  --ccccccccccccccccccccc  "MISC"                                            \
  --ggcm_mhd_do_badval_checks  1                                             \
  --monitor_conservation       0                                             \
  --cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc \
  2>&1 | tee ${run_name}.log

errcode=$?

# plot final time step:
# viscid_2d --slice z=0.0 -p rr -o log -p vx -p bx -t -1 ot_vlct.3d.xdmf

# plot movie:
# viscid_2d --slice z=0.0 -p rr -o log -p vx -p bx -s 5,10 \
#           -a ot_vlct.mp4 --np 2 ot_vlct.3d.xdmf

exit $errcode

##
## EOF
##
