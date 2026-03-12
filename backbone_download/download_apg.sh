#!/bin/bash
# Download APG (SelfPromptDeit) checkpoints for CI-CBM
# Source: https://drive.google.com/drive/folders/1DRpbNpkJ2lwIPtO_PF-mFV_kKIYKeHgt

cd ../
mkdir -p backbone_checkpoints
cd backbone_checkpoints

# Download the chkpts folder from Google Drive
# Contains: my_deit_B50_85.5_no_cls, my_deit_tiny_80_CIFAR100_B50_nocls, pretrained_chkpts, teachers
gdown --folder "https://drive.google.com/drive/folders/1DRpbNpkJ2lwIPtO_PF-mFV_kKIYKeHgt"
mv chkpts Models_Trained_by_APG

cd ../
echo ""
echo "Download complete!"
echo "Set MODEL_ROOTS[\"SelfPromptDeit_mytiny\"] in data_utils.py to: backbone_checkpoints/Models_Trained_by_APG"
