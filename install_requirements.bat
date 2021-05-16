@echo off
echo Installing Python requirements...
echo ---------------------------------
echo If you get an error that "pip is not a recognized command or file", make sure to install Python with PATH configured or add pip and python to PATH manually.
pip install tensorflow==2.5.0rc3 mediapipe==0.8.4.2 numpy==1.19.5 opencv-python==4.5.2.52
pause