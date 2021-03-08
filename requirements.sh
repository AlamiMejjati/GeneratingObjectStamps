
eval "$(conda shell.bash hook)"
conda create --name stamps python=3.7
conda activate stamps
conda install -c anaconda tensorflow-gpu=1.13.1
conda install -c conda-forge pycocotools
pip install tensorpack==0.9.4
pip install opencv-python==4.1.0.25
conda install -c anaconda pillow