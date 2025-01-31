# Install git
apt-get install git tmux -y
# Install uv
curl -LsSf https://astral.sh/uv/0.5.18/install.sh | sh
source $HOME/.local/bin/env

# clone repo
git clone https://Percival33:<TOKEN>@github.com/Percival33/phish-target-recognition.git

# checkout
cd phish-target-recognition && git checkout VP-script1

# Download processed dataset
uv run --with gdown gdown 1ewejN6qo3Bkb8IYSKeklU4GIlRHqPlUC -O - --quiet | tar zxvf - -C ../../../data/interim

# Login to wandb
uv run wandb login

# Run training
uv run trainer_phase2.py
