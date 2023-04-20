# KSB

A character recognition project based on python and PyTorch

## 主要技术栈
|  项目   | 描述  |
|  ----  | ----  |
| torch:2.0.0+cu118  | 深度学习框架 |
| numpy:1.21.1  | 数据计算库 |
| opencv-python:4.7.0.72 | 图像处理库 |
| matplotlib:3.7.1 | 画图库 |
| PyQt5:5.15.9 | ui设计 |
| tensorboard:2.12.2 | 数据可视化工具 |

## 启动
### step 1:
> pip install -r requirements.txt

注意:由于requirements.txt是本地自动生成的,所包含的库较多,可自行根据需要安装

### step 2:
切换到项目目录
> python build.py

### step 3:
这里有两个训练模块 train_a-z_0-9.py 和 train_exaple.py
区别是 train_a-z_0-9.py训练的模块能识别字母A-Z和数字0-9, 而train_exaple.py训练的模块只能识别数字0-9
这里以 train_a-z_0-9.py举例
首先运行命令 
> pyhton train_a-z_0-9.py

输出的模型在'output/AZ'目录下

### step 4:
预测识别字符
在pre.py文件中定义了两个函数tes_model和test_AZ_model,你可以直接使用他们来预测数据
> python pre.py
