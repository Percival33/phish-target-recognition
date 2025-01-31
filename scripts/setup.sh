# Install git
apt-get install git
# Install uv
curl -LsSf https://astral.sh/uv/0.5.18/install.sh | sh
source $HOME/.local/bin/env

# Download processed dataset
uv run --with gdown gdown 1ewejN6qo3Bkb8IYSKeklU4GIlRHqPlUC -O - --quiet | tar zxvf - -C ?FOLDER_PATH?

# Run training
uv run trainer_phase2.py
