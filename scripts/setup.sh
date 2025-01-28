# Install git
apt-get install git
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env


wget -O VisualPhish.zip <URL> && unzip VisualPhish.zip

# Download models
uv run --with gdown gdown 1dNi4zoU5x0N6fAMLLyEfN1vkddFkxizn -O - --quiet | tar zxvf -

# Run training
uv run trainer_phase2.py
