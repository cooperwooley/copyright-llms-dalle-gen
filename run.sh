# run.sh


# create virtual env
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# install dependencies
echo "Installing dependenices..."
pip install --upgrade pip
pip install -r requirements.txt

# run script
echo "Running DALLE Image Generation script..."
python3 generate_images.py