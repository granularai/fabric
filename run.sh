echo 'starting dependency install...'
pip install -r requirements.txt
echo 'completed dependency install.'
echo 'starting training...'
python train.py
