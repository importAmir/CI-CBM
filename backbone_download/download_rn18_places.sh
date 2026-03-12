cd ../
mkdir -p backbone_checkpoints
cd backbone_checkpoints
gdown https://drive.google.com/u/0/uc?id=1HzBdFrmzmsJX3lNJP5JRlAY3ruMr76Zz -O resnet18_places365.pth.tar
cd ..
echo ""
echo "Download complete!"
echo "Set MODEL_ROOTS[\"resnet18_places365\"] in data_utils.py to: backbone_checkpoints/resnet18_places365.pth.tar"