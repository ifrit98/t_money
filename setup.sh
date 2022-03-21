# TODO: make sure this works
source ~/anaconda3/bin/activate;
conda create --yes --name FastStyleTransfer python=3.9;
conda activate FastStyleTransfer;
pip install --upgrade pip;
pip install tensorflow matplotlib tensorflow_hub;