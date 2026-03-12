cd ../
mkdir -p backbone_checkpoints
cd backbone_checkpoints
gdown --id 1lXOj2d0_lZgPmq9NQrm-_MU0LWfmEBLI
mkdir -p Models_Trained_by_FeTrIL
unzip -o Models_Trained_by_FeTrIL.zip -d Models_Trained_by_FeTrIL
# Flatten if zip had a root folder with same name
if [ -d Models_Trained_by_FeTrIL/Models_Trained_by_FeTrIL ]; then
  mv Models_Trained_by_FeTrIL/Models_Trained_by_FeTrIL/* Models_Trained_by_FeTrIL/
  rmdir Models_Trained_by_FeTrIL/Models_Trained_by_FeTrIL 2>/dev/null || true
fi
rm -f Models_Trained_by_FeTrIL.zip
cd ..
echo ""
echo "Download complete!"
echo "Set MODEL_ROOTS[\"resnet18_FeTrIL\"] in data_utils.py to: backbone_checkpoints/Models_Trained_by_FeTrIL"