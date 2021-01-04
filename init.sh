<<comment
echo
echo 'Setting up a new virtual environment...'
echo
echo y | conda create -n rlassist python=3.8
echo
#sudo apt install gcc
conda activate rlassist
pip install --upgrade pip
pip install scipy psutil regex cython unqlite tensorflow tf_slim

mkdir logs
mkdir data/network_inputs
mkdir data/checkpoints
echo 'done!'

echo
echo 'Downloading DeepFix dataset...'
wget https://www.cse.iitk.ac.in/users/karkare/prutor/prutor-deepfix-09-12-2017.zip -P data/
cd data
unzip prutor-deepfix-09-12-2017.zip
mv prutor-deepfix-09-12-2017/* iitk-dataset/
rm -rf prutor-deepfix-09-12-2017 prutor-deepfix-09-12-2017.zip
cd iitk-dataset/
gunzip prutor-deepfix-09-12-2017.db.gz
mv prutor-deepfix-09-12-2017.db dataset.db
cd ../..
comment

export PYTHONPATH=.
echo 'Preprocessing DeepFix dataset...'
python data_processing/preprocess.py
echo 'Generating training and validation dataset...'
python -O data_processing/training_data_generator.py
echo 'Converting DeepFix dataset to RLAssist format...'
python -O data_processing/deepfix_to_rlassist_test_data_converter.py