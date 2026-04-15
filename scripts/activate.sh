# Source this instead of .venv/bin/activate to pin BLAS threads for pyspi.
#   usage: source scripts/activate.sh
#
# Pinning to 1 prevents oversubscription when PYSPI_N_JOBS > 1 (each worker
# would otherwise spawn N BLAS threads). Scoped to this shell only — does not
# affect sklearn/torch work done in other venvs or the base conda env.

_here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
source "$_here/.venv/bin/activate"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

unset _here
