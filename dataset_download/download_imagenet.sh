# Follow https://cloud.google.com/tpu/docs/imagenet-setup#verify-space

mkdir imagenet
mkdir imagenet/train
mkdir imagenet/validation

nohup wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
tar xf ./ILSVRC2012_img_train.tar -C ./train

cd ./train

for f in *.tar; do
  d=`basename $f .tar`
  mkdir $d
  tar xf $f -C $d
  rm $f  # Deletes the tar file after extraction
done

wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train_t3.tar
tar xf ./ILSVRC2012_img_train_t3.tar -C ./train

cd `$IMAGENET_HOME/train`

for f in *.tar; do
 d=`basename $f .tar`
 mkdir $d
 tar xf $f -C $d
 rm $f  # Deletes the tar file after extraction
done

# n02085620

wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
tar xf $IMAGENET_HOME/ILSVRC2012_img_val.tar -C $IMAGENET_HOME/validation

wget -O $IMAGENET_HOME/synset_labels.txt \
https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_2012_validation_synset_labels.txt