Step by Step guide to start training.

First clone the repository:
  git clone https://github.com/Strong-AI-Lab/Neuromodulated-Transformer-with-External-Memory what-you-want-it-to-be-called-on-your-machine

The following three files need to be downloaded and put in the folder `datasets'.
  https://drive.google.com/file/d/1zDtp3MIR1oHSoI1v3MVJ9eHxooSshuyv/view?usp=drive_web
  
  https://drive.google.com/file/d/17F-UzQHvPLoKkD36tzRW4mWhMs_YU86p/view?usp=drive_web
  
  https://drive.google.com/file/d/1oLCyxK3nP8wBGwwhUWDEmwXr0x86gGeV/view?usp=drive_web

The file to execute training is called `train_nmdec_wikitext.py'
  python3 train_nmdec_wikitext.py or python train_nmdec_wikitext.py to start training.
    
The current file is currently set up to perform two epochs of training on the train dataset only.

Line 3: Change the integer to gpu's device number that you wish to train on.
Line 78: batch size of 4 is used, this results in crashing on server 1, hopefully not on server 2.
Line 138: This is irrelevant as we are loading from already processed files.
Line 141-143 require the folder above to be downlaoded and put into the datasets folder (or change the directory to where you put them on your machine.
