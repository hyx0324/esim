# ESIM
tensorflow实现的ESIM

论文地址：https://arxiv.org/abs/1609.06038

参考了https://github.com/terrifyzhao/text_matching

但其中的逻辑和论文有些出入，我在此基础上做了一些修改

我的环境为：
+ tensorflow = 1.15
+ python = 3.6

input文件夹中有训练数据的demo，格式为tsv。因为我使用的是公司数据，就不公开了。

训练运行`trian.py`，预测运行`predict.py`

后续有时间的话会更新estimator版本
