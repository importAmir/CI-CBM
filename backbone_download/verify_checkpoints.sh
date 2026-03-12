#!/bin/bash
# Verify that all backbone checkpoints are downloaded and available.
# Run from project root or backbone_download.

cd "$(dirname "$0")/.."
BASE="backbone_checkpoints"

ok=0
fail=0

check() {
  if [ -e "$1" ]; then
    echo "OK: $1"
    ((ok++))
  else
    echo "MISSING: $1"
    ((fail++))
  fi
}

echo "Checking backbone checkpoints..."
echo ""

check "$BASE/resnet18_places365.pth.tar"
check "$BASE/Models_Trained_by_FeTrIL/cifar100/seed1993/b50/scratch.pth"
check "$BASE/Models_Trained_by_FeTrIL/imagenetsubset/seed1993/b50/scratch.pth"
check "$BASE/Models_Trained_by_APG/my_deit_B50_85.5_no_cls/net_0_task_0.pth"
check "$BASE/Models_Trained_by_APG/my_deit_tiny_80_CIFAR100_B50_nocls/net_0_task_0.pth"

echo ""
if [ $fail -eq 0 ]; then
  echo "All checkpoints available."
else
  echo "$ok found, $fail missing. Run the download scripts in backbone_download/."
  exit 1
fi
