sudo apt-get update
sudo apt-get install -y locales
sudo locale-gen en_US.UTF-8
sudo update-locale LANG=en_US.UTF-8
export LANG=en_US.UTF-8
sudo apt-get update
sudo apt-get install -y \
    git \
    tmux

# Install uv
curl -LsSf https://astral.sh/uv/0.5.18/install.sh | sh
source $HOME/.local/bin/env

# clone repo
git clone https://Percival33:<TOKEN>@github.com/Percival33/phish-target-recognition.git

# checkout
cd phish-target-recognition && git checkout VP-script1

# Download processed dataset
uv run --with gdown gdown 1ewejN6qo3Bkb8IYSKeklU4GIlRHqPlUC -O - --quiet | tar zxvf - -C ../../../data/interim

# smallerSampleDataset
#uv run --with gdown gdown 1agHs15tIlKvXqC8M65ZXln2mEDPz6fSk -O - --quiet | tar zxvf - -C ../../../data/interim

# vp original paper results on smaller dataset
uv run --with gdown gdown 1xPN_3GYo8tcdnCPwPdKXzUS13-K54NLg -O - --quiet | tar zxvf - -C ../../../data/interim

# run gaptb94y
uv run --with gdown gdown 1rhXI6CluYxizS1mEEItvCtAVJJInr4Up -O - --quiet | tar zxvf - -C ../../../data/interim

# Login to wandb
uv run wandb login

# Run training
#uv run trainer_phase2.py

# smaller sample dataset
#uv run trainer.py --num-targets 5 --legit-imgs-num 420 --phish-imgs-num 160 --save-interval 20 --n-iter 20 --num-sets 5 --hard-n-iter 5 --num-sets 5 --lr-interval 250

# tar --disable-copyfile --no-xattrs -czvf archive.tar.gz folder_name