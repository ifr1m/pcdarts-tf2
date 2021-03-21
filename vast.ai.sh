apt install -qq -y python3.8 wget htop google-perftools &&
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1 &&
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2 &&
update-alternatives --config python3 &&
python -m pip install --upgrade pip ;
pip install -r requirements.txt ;
wget https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py -O vast; chmod +x vast;
cp vast /usr/local/bin