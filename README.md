# FogHaze

1/ Perlin noise library
https://github.com/caseman/noise

Installation uses the standard Python distutils regime:

python setup.py install

This will compile and install the noise package into your Python site
packages.

The functions and their signatures are documented in their respective
docstrings.  Use the Python help() function to read them.

>>> import noise 
>>> help(noise)

1/ Các dependencies:
[ ] numpy
[ ] opencv-contrib-python
[ ] scikit
[ ] matplotlib
[ ] pandas
[ ] caseman/noise
[ ] torch
[ ] nvidia
[ ] scikit-image
[ ] scipy

2/ Giải thích một số thư mục và files quan trọng
foghaze_generator/: logic 
thuật toán sinh sương mù
foghaze_removal/: logic thuật toán lọc bỏ sương mù

interactive_fh_generator: GUI đơn giản thử nghiệm các tham số của thuật toán tạo sương mù
defoghaze.py: chương trình console cho thuật toán lọc bỏ sương mù
