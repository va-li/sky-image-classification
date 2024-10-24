echo "-- Setting up Raspberry Pi --"

echo "-- Updating and upgrading packages --"
sudo apt update
sudo apt upgrade -y

echo "-- Installing System Packages --"
sudo apt install python3-pip -y
sudo apt install tmux -y

# set up tmux
cat <<EOF > ~/.tmux.conf
# reload config file
bind r source-file ~/.tmux.conf

# remap prefix from 'C-b' to 'C-a'
unbind C-b
set-option -g prefix C-a
bind-key C-a send-prefix

# split panes using | and -
bind | split-window -h
bind - split-window -v
unbind '"'
unbind %

# switch panes using Alt-arrow without prefix
bind -n M-Left select-pane -L
bind -n M-Right select-pane -R
bind -n M-Up select-pane -U
bind -n M-Down select-pane -D

# enable mouse mode
set -g mouse on

# bash will be the default shell
set-option -g default-command bash
EOF

echo "-- Creating folders and virtual environment --"

mkdir -p ~/sky-image-classification
cd ~/sky-image-classification

# create virtualenv with python 3.11
python3.11 -m venv venv --python=python3.11
source venv/bin/activate

echo "-- Installing Python packages --"
# upgrade pip
pip install --upgrade pip
# install packages for image classification
pip install torch torchvision torchaudio
pip install install opencv-python-headless
pip install numpy --upgrade
