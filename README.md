# Dolphin Attack-Detection

特征提取：

双声道合并为单声道，做MFCC，mel滤波器组共40个滤波器，分帧参数：帧长2048个点，每次移动512个点，采样频率48kHz。

做long-time average MFCC（自己拍脑袋想出来的），就是把所有帧的同一个梅尔滤波器输出的结果做一个average，优点如下：

* 可以充分利用整个音频文件的数据
* 消除音频时间长短不一对SVM的影响
* 避免上、下采样（即通过增加或减少采样频率（抽样、插值）把音频的时间调整至一致）带来的频谱混叠或者失真
* 降低运算量，40个mel滤波器输出后feature长度仅为40

可能缺陷：

* 特征数量太少，容易导致识别效果不佳

![clip_image001.png](https://i.loli.net/2019/12/28/Xv7wiHZnt6UO31k.png)

整段语音直接FFT再挑500Hz-1000Hz之间的频段作为feature：

效果很差（模型直接过拟合，在测试集上跑时全部判断sample是demod类型的），原因可能在于：

1.  每个sample的长度不一样，虽然可以补零再FFT，但是这样会造成feature的长度太长，容易导致SVM过拟合

2.  即使通过Maxpool或其他方法来提取FFT的包络并且减少feature的长度，但语音信号是非平稳信号，不满足FFT的条件，而且FFT的特征会受到说话内容的影响，从而导致SVM难以将其分类。






二分类效果衡量标准：https://www.cnblogs.com/futurehau/p/6109772.html

![clip_image003.jpg](https://i.loli.net/2019/12/28/VZ6BtwAExC4p2sI.jpg)

Positive_acc = TP / (TP + FN)

Nagetive_acc = TN / (FP + TN)

All_acc = (TP + TN) / (TP + FN + FP + TN)

 

RBF参数选择：

https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html#sphx-glr-auto-examples-svm-plot-rbf-parameters-py

主要是C和Gamma两个超参数

 

对线性核、RBF核做Grid search，结合Cross validation

Cross validation: https://zhuanlan.zhihu.com/p/24825503

Grid search: 暴力搜索

结果：

![clip_image007.png](https://i.loli.net/2019/12/28/UCGVeRQYDW1kxzu.png)

![clip_image005.png](https://i.loli.net/2019/12/28/Uk1oNEyPupwsvh6.png)

总之出来的结果就是很好就是了，基本上C随便取一取，然后gamma往小了调就行。

