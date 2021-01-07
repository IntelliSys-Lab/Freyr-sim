#! /bin/bash

#
# Activate pytorch_latest_p37 permanently on g4dn.2xlarge
#

source activate pytorch_latest_p37

echo "" >> ~/.bashrc
echo "# Activate pytorch_latest_p37" >> ~/.bashrc
echo "source activate pytorch_latest_p37" >> ~/.bashrc
echo "" >> ~/.bashrc

#
# Create ckpt, figures, logs and azure trace folders if not exist
#

folders="./ckpt ./figures ./logs ./azurefunctions-dataset2019"

for folder in $folders
do
    if [ ! -d "$folder" ]
    then
        mkdir $folder
    # else
    #     rm $folder/*
    fi
done

#
# Download Azure Functions traces
#

cd ./azurefunctions-dataset2019
wget "https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/azurefunctions_dataset2019/azurefunctions-dataset2019.tar.xz"
tar -xvf azurefunctions-dataset2019.tar.xz
cd ..
