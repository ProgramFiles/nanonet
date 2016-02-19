A: 
    To get the code running, you would need to setup a GPU instacne on AWS, please follow my ansible example:
    https://git/algorithm/ranger
    Remove the section under #Setup /etc/fstab, as your instance will not have access to my storage server. 

B:
    If you wish to setup the AWS instance by yourself, here is the key requirment besides the GPU driver:
    1. currennt software itself, must be complied and avaliavle under /home/ubuntu/bin/currennt
    2. these common libs: 
            - h5utils
            - hdf5-tools
            - hdf5-helpers
            - libhdf5-dev
            - libhdf5-serial-dev
            - python-h5py
            - libboost-all-dev 
            - libnetcdf-dev 
            - unzip
            - netcdf-bin
    3. netcdf4-python: https://github.com/Unidata/netcdf4-python.git 

C:
    Some basics of how the trainning should be done is here:
    https://wiki.oxfordnanolabs.local/display/RES/Best+practice+for+using+CURRENNT+on+new+pores+and+motors

D:
    A few ANN structure config files are located under data/ 
    Note that, these configs are based on a specific input window size. Thus if you change the window size, the size of the first layer much also be changed accordingly. 
    As Tim said, tuning the ANN to get optimal perfermence for a specific pore is the resposiblity of data analysis. Thus any ANN configs included here only serves as a starting point and by no mean optimal.  

E:
    For a test run, try:
    python ann_trainer.py --train example/train.nc --val example/val.nc --output example/test --model data/L1R1N2.jsn

F:
    From basecalled mapped fast5 files to ANN basecall:
    1. copy all the fast5 files into one folder
    2. run prepare_training_files.py to split fast5 files into trainning, validate, and testing. This step will give you 3 files containing the file paths to fast5 files.
    3. run ann_trainer.py with --fast5_list to train the ANN, typicall training time with 5 million data points, uisng the example ANN configs in data/, should be < 48 hours.
    4. after the long wait, run ann_basecaller.py, using the trained ANN to basecall, remember to limit the read length, 1000 reads should take < 30 min  
