# !/usr/bin/bash
# This script downloads the COCO2014 captions and its correspond bottom-up features.

# captions
echo "Downloading COCO captions ... "
wget -O ./data/caption_datasets.zip https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip

echo "Unzipping the captions ... "
unzip ./data/caption_datasets.zip -d ./data

rm -f ./data/caption_datasets.zip

# features
echo "Downloading bottom-up features ... "
wget -O ./data/trainval.zip https://storage.googleapis.com/up-down-attention/trainval.zip

echo "Unzipping the features"
unzip ./data/trainval.zip -d ./data

rm -f ./data/trainval.zip
rm -rf ./data/trainval
