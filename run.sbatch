#!/bin/bash
#SBATCH --job-name=mv_distill
#SBATCH --partition=dheld
#SBATCH --nodelist=grogu-4-23
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/project_data/held/peilin/svm/logs/mv_distill/%x-%j.out
#
# Run one mv_distill CLI on grogu-4-23 (the student needs 24.5 GiB):
#   sbatch run.sbatch mv_distill.train_student --teacher teacher --run student
#   sbatch -J label run.sbatch mv_distill.label --teacher teacher --dataset scan
#   sbatch --gres=gpu:2 run.sbatch mv_distill.build_wrist fill ...   (one process per GPU: see README)
# Data stays on the node's /scratch (MV_DISTILL_RAW / MV_DISTILL_WORK, overridable from the
# submitting shell); this folder on /project_data only holds code, checkpoints and logs.

set -eu
export MV_DISTILL_RAW=${MV_DISTILL_RAW:-/scratch/pmeng/mv_distill/raw}
export MV_DISTILL_WORK=${MV_DISTILL_WORK:-/scratch/pmeng/mv_distill/work}
cd /project_data/held/peilin/svm/mv_distill
echo "[$(date)] $(hostname): python -m $*"
~/miniconda3/envs/gap/bin/python -m "$@"
echo "[$(date)] done"
