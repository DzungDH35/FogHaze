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

3/ Giải thích về chương trình demo thuật toán sinh sương mù (console)
- Ảnh đầu vào (bắt buộc)
- Depth map (optional). Nếu không cung cấp, sẽ sử dụng Midas
-  Operation mode:
   +  atmospheric light: naive_int --> random ra 1 int, naive_arr --> random ra 1 mảng atm
   +  scattering coef: naive_int --> random ra 1 int, naive_arr --> random ra 1 mảng, pnoise --> sử dụng nhiễu Perlin với cấu hình riêng
- Configure params:
  + Nếu cung cáp A hoặc beta thì sẽ sử dụng chúng, không thì tùy thuộc vào operation mode
  + Các tham số octaves, v.v là của Perlin noise
