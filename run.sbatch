#!/bin/bash
#SBATCH --job-name=mv_distill
#SBATCH --partition=dheld
#SBATCH --nodelist=grogu-4-23
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=4-00:00:00
#SBATCH --output=/project_data/held/peilin/svm/logs/mv_distill/%x-%j.out
#
# Run one mv_distill CLI on grogu-4-23 (the staged tasks are on its /scratch):
#   sbatch run.sbatch mv_distill.train_student --run student2
#   sbatch -J infer run.sbatch mv_distill.infer --run student2
#   sbatch -J centres --gres=gpu:0 -c 32 run.sbatch mv_distill.build_centres build --workers 32   (CPU only)
#   NPROC=2 sbatch --gres=gpu:2 -c 32 --mem=192G run.sbatch mv_distill.train_student --arch qwen3d --run qwen3d
#     (NPROC > 1: one process per GPU with RANK / LOCAL_RANK / WORLD_SIZE set, as torchrun does;
#      torchrun itself takes our --run for an abbreviation of its --run-path)
# Data stays on the node's /scratch (MV_DISTILL_DATA / MV_DISTILL_WORK, overridable from the
# submitting shell); this folder on /project_data only holds code, checkpoints and logs.
# The python is the qwen3d venv: the gap env + transformers 4.51.1 (for the vendored Qwen-3D).

set -eu
export MV_DISTILL_DATA=${MV_DISTILL_DATA:-/scratch/pmeng/svm_fm:/scratch/pmeng/svm_fm_rest}
export MV_DISTILL_WORK=${MV_DISTILL_WORK:-/scratch/pmeng/mv_distill/work}
export WANDB_DIR=${WANDB_DIR:-/project_data/held/peilin/svm/logs}   # offline runs: wandb sync logs/wandb/offline-run-*
export WANDB_MODE=${WANDB_MODE:-offline}
export PYTHONUNBUFFERED=1   # log lines reach the .out file as they are printed
cd /project_data/held/peilin/svm/mv_distill
PY=${PY:-$HOME/venvs/qwen3d/bin/python}
NPROC=${NPROC:-1}
echo "[$(date)] $(hostname): python -m $* (data $MV_DISTILL_DATA, $NPROC proc)"
if [ "$NPROC" -gt 1 ]; then
  export MASTER_ADDR=127.0.0.1 MASTER_PORT=$((20000 + ${SLURM_JOB_ID:-0} % 20000)) WORLD_SIZE=$NPROC
  for ((r = 0; r < NPROC; r++)); do RANK=$r LOCAL_RANK=$r $PY -m "$@" & done
  status=0
  for ((r = 0; r < NPROC; r++)); do    # the first failure stops the other ranks
    wait -n || { status=$?; kill $(jobs -p) 2>/dev/null || true; }
  done
  [ "$status" -eq 0 ] || exit "$status"
else
  $PY -m "$@"
fi
echo "[$(date)] done"
