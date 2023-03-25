# import os
# from matplotlib import pyplot as plt

# os.add_dll_directory(r'D:\Softwares\Program_Files\C\mingw64\bin')
# os.add_dll_directory(r'D:\Softwares\Program_Files\C\cpplibs\opencv\install\x64\mingw\bin')
# import transform
# transformer = transform.Transform()
# img = transformer.radon_transform_with_noise(
#     r"D:\Code\pycode\graduate_design\data\imgs\1.png",
#     1.5
# )
# plt.imshow(img, cmap='gray')
# plt.show()
from src import graduate_design
graduate_design.generate_data()
