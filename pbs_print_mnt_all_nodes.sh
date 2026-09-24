#!/bin/bash
#PBS -N print-rdata4-3-all-nodes
#PBS -l select=1:ncpus=1:mem=2gb
#PBS -l walltime=00:10:00

set -euo pipefail

PROJECT_DIR="/home/ramanathan/VLM/lmms-eval"
TARGET_DIR="/mnt/rdata4_3"
DEFAULT_RESULT_DIR="$PROJECT_DIR/outputs/rdata4_3_contents_by_node"
DEFAULT_NODES="cvml03 cvml04 cvml05 cvml06 cvml07 cvml10 cvml11 cvml12"

submit_workers() {
  local result_dir="${RESULT_DIR:-$DEFAULT_RESULT_DIR}"
  local node_list="${NODES:-$DEFAULT_NODES}"
  local node

  mkdir -p "$result_dir"

  echo "Submitting one $TARGET_DIR inspection job to each node: $node_list"
  for node in $node_list; do
    qsub \
      -N "rdata4-3-${node}" \
      -l "select=1:ncpus=1:mem=2gb:host=${node}" \
      -o "$result_dir/${node}.pbs.out" \
      -e "$result_dir/${node}.pbs.err" \
      -v "MNT_SCAN_WORKER=1,TARGET_HOST=${node},RESULT_DIR=${result_dir}" \
      "$PROJECT_DIR/pbs_print_mnt_all_nodes.sh"
  done

  echo "Worker results will be written to: $result_dir"
  echo "After all jobs finish, print them with:"
  echo "  cat $result_dir/cvml*.txt"
}

inspect_node() {
  local requested_host="${TARGET_HOST:?TARGET_HOST was not supplied by the dispatcher}"
  local result_dir="${RESULT_DIR:-$DEFAULT_RESULT_DIR}"
  local actual_host
  local temporary_result

  actual_host="$(hostname -s)"
  mkdir -p "$result_dir"
  temporary_result="$(mktemp "$result_dir/.${requested_host}.XXXXXX")"
  trap 'rm -f "$temporary_result"' EXIT

  {
    echo "requested_host=$requested_host"
    echo "actual_host=$actual_host"
    echo "pbs_job_id=${PBS_JOBID:-not-set}"
    echo "timestamp=$(date --iso-8601=seconds)"

    echo
    echo "[ls -la $TARGET_DIR]"
    ls -la --time-style=long-iso "$TARGET_DIR" 2>&1

    echo
    echo "[immediate entries under $TARGET_DIR]"
    find "$TARGET_DIR" -mindepth 1 -maxdepth 1 \
      -printf '%y\t%p\t%l\n' 2>&1 | sort

    echo
    echo "[mounted filesystems at or below $TARGET_DIR]"
    if command -v findmnt >/dev/null 2>&1; then
      findmnt -rn -o TARGET,SOURCE,FSTYPE,OPTIONS 2>&1 \
        | awk -v target="$TARGET_DIR" '$1 == target || index($1, target "/") == 1'
    else
      echo "findmnt is unavailable"
    fi
  } > "$temporary_result"

  mv "$temporary_result" "$result_dir/${requested_host}.txt"
  trap - EXIT

  cat "$result_dir/${requested_host}.txt"
}

if [[ "${MNT_SCAN_WORKER:-0}" == "1" ]]; then
  inspect_node
else
  submit_workers
fi
