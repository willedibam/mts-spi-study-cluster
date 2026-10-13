# Sourced by every Gadi launcher and submitter: modules, the environment, one
# thread per numerical library, and the project's storage roots. Run from the
# repository root.
module purge
module load python3/3.12.1 nci-parallel/1.0.0a
source .venv/bin/activate

# pyspi parallelises over SPIs with processes; threaded libraries under those
# workers time-share the cores. The runners refuse several workers without this.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# Configs may write these as $MTS_SCRATCH and $MTS_GDATA instead of full paths.
export MTS_SCRATCH=/scratch/ql44/we2614/mts-spi-study
export MTS_GDATA=/g/data/ql44/we2614/mts-spi-study

# Optional pyspi source snapshot (a development branch) ahead of the shared
# editable install; meta.json records which pyspi ran.
if [[ -n "${PYSPI_SOURCE:-}" ]]; then
    test -d "$PYSPI_SOURCE/pyspi"
    export PYTHONPATH="$PYSPI_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
fi
