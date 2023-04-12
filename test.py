# import os
# from matplotlib import pyplot as plt

# os.add_dll_directory(r'D:\Softwares\Program_Files\C\mingw64\bin')
# os.add_dll_directory(r'D:\Softwares\Program_Files\C\cpplibs\opencv\install\shared\x64\mingw\bin')
# import cpptrans
# img = cpptrans.radon_transform_with_noise(
#     r"D:\Code\py\graduate_design\data\120x120_100_255_10_3_[0, 180]_1_without_noise\imgs\1.png",
#     1.5
# )
# plt.imshow(img, cmap='gray')
# plt.show()
from src import graduate_design
graduate_design.generate_data()
