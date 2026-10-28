#!/bin/bash
#SBATCH --job-name=mv_teachers
#SBATCH --partition=dheld
#SBATCH --nodelist=grogu-4-23
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/project_data/held/peilin/svm/logs/mv_distill/%x-%j.out
#
# The per-task teachers, one after another on one GPU: train_teacher, then label every
# frame of every staged root. Runs are <TEACHER>_<task> (TEACHER default teacher2).
#   sbatch teachers.sbatch handover_block hanging_mug [more tasks ...]

set -eu
export MV_DISTILL_DATA=${MV_DISTILL_DATA:-/scratch/pmeng/svm_fm:/scratch/pmeng/svm_fm_rest}
export MV_DISTILL_WORK=${MV_DISTILL_WORK:-/scratch/pmeng/mv_distill/work}
export WANDB_DIR=${WANDB_DIR:-/project_data/held/peilin/svm/logs}   # offline runs: wandb sync logs/wandb/offline-run-*
export WANDB_MODE=${WANDB_MODE:-offline}
TEACHER=${TEACHER:-teacher2}
cd /project_data/held/peilin/svm/mv_distill
PY=${PY:-$HOME/venvs/qwen3d/bin/python}
for t in "$@"; do
  echo "[$(date)] $(hostname): teacher $t -> runs/${TEACHER}_$t (data $MV_DISTILL_DATA)"
  $PY -m mv_distill.train_teacher --task "$t" --run "${TEACHER}_$t"
  $PY -m mv_distill.label --task "$t" --teacher "${TEACHER}_$t"
done
echo "[$(date)] done"
