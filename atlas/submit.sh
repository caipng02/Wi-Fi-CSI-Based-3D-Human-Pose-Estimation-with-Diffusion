#!/bin/bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
mode="${1:-all}"
data="${2:-/hpctmp/$USER/fyp/MMFi_Dataset}"
queue="${QUEUE:-parallel}"
run="${RUN_DIR:-$repo/runs/atlas_$(date +%Y%m%d_%H%M%S)}"
encoder="${ENCODER_CHECKPOINT:-$run/shared_encoder.pt}"

case "$mode" in
    all|pretrain|baseline|grc|test_baseline|test_grc) ;;
    *) echo "Use all, pretrain, baseline, grc, test_baseline or test_grc." >&2; exit 2 ;;
esac
if [[ "$mode" == all && -n "${RESUME_CHECKPOINT:-}" ]]; then
    echo "Choose one component to resume, such as grc or pretrain." >&2
    exit 2
fi
for value in "$data" "$run" "$encoder" "${RESUME_CHECKPOINT:-}" \
             "${POSE_CHECKPOINT:-}" "${ATLAS_ENV_FILE:-$repo/atlas/environment.sh}"; do
    if [[ "$value" == *,* || "$value" == *$'\n'* ]]; then
        echo "PBS -v paths must exclude commas and newlines: $value" >&2
        exit 2
    fi
done
test -d "$data/E01"
test -r atlas/phase1.pbs
test -r "${ATLAS_ENV_FILE:-$repo/atlas/environment.sh}"
command -v qsub >/dev/null
mkdir -p "$run"
jobs="$run/pbs_jobs.tsv"
if [[ -e "$jobs" ]]; then
    echo "Choose a fresh RUN_DIR. Job IDs already exist at $jobs" >&2
    exit 2
fi
printf 'stage\tjob_id\n' > "$jobs"
echo "Run folder: $run"

submit() {
    local stage=$1 walltime=$2
    shift 2
    local variables="STAGE=$stage,DATA_ROOT=$data,RUN_DIR=$run,ENCODER_CHECKPOINT=$encoder"
    variables+=",END_EPOCH=${END_EPOCH:-30},DEVICE=${DEVICE:-cpu},WORKERS=${WORKERS:-2}"
    variables+=",RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-},POSE_CHECKPOINT=${POSE_CHECKPOINT:-}"
    variables+=",ATLAS_ENV_FILE=${ATLAS_ENV_FILE:-$repo/atlas/environment.sh}"
    variables+=",OMP_NUM_THREADS=8,MKL_NUM_THREADS=8,OPENBLAS_NUM_THREADS=8"
    job_id=$(qsub -q "$queue" -N "fyp_${stage}" \
        -l "select=${SELECT:-1:ncpus=12:mem=32gb}" -l "walltime=$walltime" \
        -v "$variables" "$@" atlas/phase1.pbs)
    printf '%s\t%s\n' "$stage" "$job_id" | tee -a "$jobs"
}

if [[ "$mode" == all ]]; then
    submit pretrain "${PRETRAIN_WALLTIME:-72:00:00}"
    pretrain_id=$job_id
    submit baseline "${TRAIN_WALLTIME:-48:00:00}" -W "depend=afterok:$pretrain_id"
    baseline_id=$job_id
    submit grc "${TRAIN_WALLTIME:-48:00:00}" -W "depend=afterok:$pretrain_id"
    grc_id=$job_id
    submit test_baseline "${TEST_WALLTIME:-12:00:00}" -W "depend=afterok:$baseline_id"
    submit test_grc "${TEST_WALLTIME:-12:00:00}" -W "depend=afterok:$grc_id"
else
    case "$mode" in
        pretrain) walltime="${PRETRAIN_WALLTIME:-72:00:00}" ;;
        test_*) walltime="${TEST_WALLTIME:-12:00:00}" ;;
        *) walltime="${TRAIN_WALLTIME:-48:00:00}" ;;
    esac
    submit "$mode" "$walltime"
fi
