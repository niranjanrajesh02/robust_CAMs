#!/bin/bash
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem 20G
#SBATCH -t 1-23:59 # time (D-HH:MM)
#SBATCH --job-name="extract_imagenet"
#SBATCH -o extract_out.log
#SBATCH -e extract_err.log
#SBATCH --cpus-per-task=10



cd /scratch/venkat/niranjan/imagenet
#
# Extract the training data:
#
# Create train directory; move .tar file; change directory

mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
# Extract training set; remove compressed file
tar -xvf ILSVRC2012_img_train.tar && mv ILSVRC2012_img_train.tar ../ 
#
# At this stage imagenet/train will contain 1000 compressed .tar files, one for each category
#
# For each .tar file: 
#   1. create directory with same name as .tar file
#   2. extract and copy contents of .tar file into directory
#   3. remove .tar file
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
#
# This results in a training directory like so:
#
#  imagenet/train/
#  ├── n01440764
#  │   ├── n01440764_10026.JPEG
#  │   ├── n01440764_10027.JPEG
#  │   ├── ......
#  ├── ......
#
# Change back to original directory
cd ../
#
# Extract the validation data and move images to subfolders:
#
# Create validation directory; move .tar file; change directory; extract validation .tar; remove compressed file
mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val 

tar -xvf ILSVRC2012_img_val.tar && mv ILSVRC2012_img_val.tar ../

# get script from soumith and run; this script creates all class directories and moves images into corresponding directories
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash