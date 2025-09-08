#!/usr/bin/env bash
set -euo pipefail

# Determine which shell rc file to write to
RC_FILE="$HOME/.zshrc"
if [ ! -f "$RC_FILE" ]; then
RC_FILE="$HOME/.bashrc"
fi

# Validate required environment variables
required_vars=(WANDB_API_KEY)
missing_vars=()
for var in "${required_vars[@]}"; do
if [ -z "${!var:-}" ]; then
missing_vars+=("$var")
fi
done
if [ "${#missing_vars[@]}" -ne 0 ]; then
echo "Missing required environment variables: ${missing_vars[*]}" >&2
exit 1
fi

apt-get update
apt-get install -y locales
locale-gen en_US.UTF-8
update-locale LANG=en_US.UTF-8
export LANG=en_US.UTF-8
apt-get update
apt-get install -y tmux \
  git \
  nvtop

# Setup uv
curl -LsSf https://astral.sh/uv/0.5.18/install.sh | sh
source "$HOME/.local/bin/env"

# Setup just
mkdir -p ~/bin
curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to ~/bin
echo 'export PATH="$PATH:$HOME/bin"' >> "$RC_FILE"
if command -v just >/dev/null 2>&1; then JUST=just; else JUST="$HOME/bin/just"; fi
$JUST --help

# Clone repo
git clone git@github.com:Percival33/phish-target-recognition.git
cd phish-target-recognition && echo "export PROJECT_ROOT_DIR=$(pwd)" >> "$RC_FILE"
$JUST tools
# Login to Wandb
uv run wandb login $WANDB_API_KEY
