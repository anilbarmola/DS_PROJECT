# Analyzing Amazon Sales data
📝 Description
This implemantation is based on official Alphapose repository https://github.com/MVIG-SJTU/AlphaPose
In this project we have used Alphapose and XGBOOST for activity recognition.
⏳ Dataset
Download the dataset for custom training
https://drive.google.com/drive/folders/1CFxvuigTzbnRXUojFeCRozxjUbYiQ8RM?usp=sharing
🏽‍ Download Object Detection Model
Download the object detection model manually : yolov3-spp.weights file from following Drive Link
https://drive.google.com/file/d/1h2g_wQ270_pckpRCHJb9K78uDf-2PsPd/view?usp=sharing
Download the weight file and Place it into " detector/yolo/data/ " folder.
🏽‍ For Pose Tracking, Download the object tracking model
For pose tracking, download the object tracking model manually: " JDE-1088x608-uncertainty " from following Drive Link
https://drive.google.com/file/d/1oeK1aj9t7pTi1u70nSIwx0qNVWvEvRrf/view?usp=sharing
Download the file and Place it into " detector/tracker/data/ ". folder.
🏽‍ Download Fast.res50.pt file
Download the " fast.res50.pth " file from following Drive Link
https://drive.google.com/file/d/1WrvycZnVWwltSa6cjeTznEFOyNAwHEZu/view?usp=sharing
Download the file and Place it into " pretrained_models/ ". folder.
🖥️ Installation
🛠️ Requirements
Python 3.5+
Cython
PyTorch 1.1+
torchvision 0.3.0+
Linux
GCC<6.0, check facebookresearch/maskrcnn-benchmark#25
⚙️ Setup
Install PyTorch :-
$ pip3 install torch==1.1.0 torchvision==0.3.0
Install :-
$ export PATH=/usr/local/cuda/bin/:$PATH
$ export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH
$ pip install cython
$ sudo apt-get install libyaml-dev
$ python setup.py build develop --user
$ python -m pip install Pillow==6.2.1
$ pip install -U PyYAML
🎯 Inference demo
Testing with Images ( Put test images in AlphaPose/examples/demo/ ) :-
$ python scripts/demo_inference.py --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/fast_res50_256x192.pth --indir examples/demo/ --save_img
Testing with Video ( Put test video in AlphaPose/examples/demo/ ) :-
$ python scripts/demo_inference.py --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/fast_res50_256x192.pth --video examples/demo/3.mp4 --outdir examples/res1 --save_video --gpus 0
📖 Please Go through Pose_With_Action_HLD2.docx for more info.
Contributors 
Developer.gif
Akshay Kumar Prasad
Akshay Namdev Kadam
