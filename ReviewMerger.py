import os
import pandas as pd

# 设置文件夹路径
folder_path = 'ReviewTotal'
all_files = os.listdir(folder_path)

# 初始化一个空的DataFrame用来存放所有文件的数据
merged_data = pd.DataFrame()

# 遍历文件夹中的所有文件
for file_name in all_files:
    print("In a new file")
    if file_name.endswith('.xlsx'):
        # 构造完整的文件路径
        file_path = os.path.join(folder_path, file_name)
        # 读取xlsx文件
        data = pd.read_excel(file_path)
        # 将读取的数据添加到merged_data中
        merged_data = pd.concat([merged_data, data], ignore_index=True)

# 将合并后的数据保存为一个新的xlsx文件
output_file_path = os.path.join(folder_path, 'merged_data.xlsx')
merged_data.to_excel(output_file_path, index=False)

print(f"All files have been merged into {output_file_path}")
