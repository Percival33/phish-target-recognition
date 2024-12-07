# check if VisualPhish directory or VisualPhish.zip exists if not download it
# wget https://oc.cs.uni-saarland.de/index.php/s/QcBJyLjn9oEiXxB/download/VisualPhish.zip

"""
#!/bin/bash

if [ ! -d "VisualPhish" ] && [ ! -f "VisualPhish.zip" ]; then
    wget https://oc.cs.uni-saarland.de/index.php/s/QcBJyLjn9oEiXxB/download/VisualPhish.zip
fi

file_ids=('1c4h9F1OjSVz8mAR0xUeH-4AzixW_l6j5' '1l29pzF1BI6KGRFGU-1IyfiWaVcC_j2PV' '1_uCJFK-gdinbblIczYEUFmlHa0c0-ALt' '1uCQWaOs2zFR1oAqbd7lZh_73N89YUaHy')

for file_id in "${file_ids[@]}"; do
    gdown "$file_id"
done

unzip VisualPhish.zip -d VisualPhish


name: visualphish
channels:
  - defaults
dependencies:
  - python=3.6
  - tensorflow-gpu=1.15.0
  - matplotlib
  - scikit-image

pip install keras==2.1.6
pip install tensorflow-gpu==1.15.0

conda activate visualphish
"""
