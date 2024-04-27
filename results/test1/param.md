# 噪声水平
noise_ratio = 0.3

# 扩散学习相关参数
vote_times = 100  # 投票次数
T = 162  # 超像素数量
pca_dim = 64  # PCA降维维度(%, dim): (0.999, 69) (0.99, 25) (0.95, 5)
unlabelled_ratio = 0.5  # unlabelled数据所占比例
alpha = 0.7  # 扩散程度

# cnn参数
epochs = 200
lr = 0.00005
batch_size = 1024


---
======================== 第1轮训练 ========================
> cnn训练:
Epoch [20/200], Train Loss: 2.0281
Epoch [40/200], Train Loss: 1.8602
Epoch [60/200], Train Loss: 1.7351
Epoch [80/200], Train Loss: 1.6478
Epoch [100/200], Train Loss: 1.5909
Epoch [120/200], Train Loss: 1.5362
Epoch [140/200], Train Loss: 1.4960
Epoch [160/200], Train Loss: 1.4542
Epoch [180/200], Train Loss: 1.4233
Epoch [200/200], Train Loss: 1.3868
Time taken[cnn]: 33.29 seconds

> 标签传播:
构建SSPTM...
预测伪标签...
Time taken[diffusion]: 49.46 seconds

> Time taken[train 1]: 82.76 seconds

> 投票结果:
Accuracy[vote:1]: 0.9185
History accuracy: [0.9185]
Total time taken: 82.76 seconds

======================== 第2轮训练 ========================
> cnn训练:
Epoch [20/200], Train Loss: 2.0301
Epoch [40/200], Train Loss: 1.8353
Epoch [60/200], Train Loss: 1.7175
Epoch [80/200], Train Loss: 1.6325
Epoch [100/200], Train Loss: 1.5715
Epoch [120/200], Train Loss: 1.5119
Epoch [140/200], Train Loss: 1.4864
Epoch [160/200], Train Loss: 1.4414
Epoch [180/200], Train Loss: 1.4175
Epoch [200/200], Train Loss: 1.3649
Time taken[cnn]: 25.87 seconds

> 标签传播:
构建SSPTM...
预测伪标签...
Time taken[diffusion]: 53.86 seconds

> Time taken[train 2]: 79.76 seconds
#debug >label_presudo_list: False

> 投票结果:
Accuracy[vote:2]: 0.9181
History accuracy: [0.9185, 0.9181]
Total time taken: 162.51 seconds

======================== 第3轮训练 ========================
> cnn训练:
Epoch [20/200], Train Loss: 1.9623
Epoch [40/200], Train Loss: 1.7903
Epoch [60/200], Train Loss: 1.6918
Epoch [80/200], Train Loss: 1.6293
Epoch [100/200], Train Loss: 1.5393
Epoch [120/200], Train Loss: 1.4924
Epoch [140/200], Train Loss: 1.4626
Epoch [160/200], Train Loss: 1.4093
Epoch [180/200], Train Loss: 1.3558
Epoch [200/200], Train Loss: 1.3293
Time taken[cnn]: 26.50 seconds

> 标签传播:
构建SSPTM...
预测伪标签...
Time taken[diffusion]: 49.78 seconds

> Time taken[train 3]: 76.30 seconds

> 投票结果:
Accuracy[vote:3]: 0.9180
History accuracy: [0.9185, 0.9181, 0.918]
Total time taken: 238.82 seconds

======================== 第4轮训练 ========================
> cnn训练:
Epoch [20/200], Train Loss: 2.0347
Epoch [40/200], Train Loss: 1.8272
Epoch [60/200], Train Loss: 1.7369
Epoch [80/200], Train Loss: 1.6599
Epoch [100/200], Train Loss: 1.5986
Epoch [120/200], Train Loss: 1.5498
Epoch [140/200], Train Loss: 1.4958
Epoch [160/200], Train Loss: 1.4637
Epoch [180/200], Train Loss: 1.4130
Epoch [200/200], Train Loss: 1.3889
Time taken[cnn]: 26.61 seconds

> 标签传播:
构建SSPTM...
预测伪标签...
Time taken[diffusion]: 49.93 seconds

> Time taken[train 4]: 76.57 seconds

> 投票结果:
Accuracy[vote:4]: 0.9182
History accuracy: [0.9185, 0.9181, 0.918, 0.9182]
Total time taken: 315.38 seconds

======================== 第5轮训练 ========================
> cnn训练:
Epoch [20/200], Train Loss: 2.0034
Epoch [40/200], Train Loss: 1.8418
Epoch [60/200], Train Loss: 1.7397
Epoch [80/200], Train Loss: 1.6623
Epoch [100/200], Train Loss: 1.5988
Epoch [120/200], Train Loss: 1.5407
Epoch [140/200], Train Loss: 1.4908
Epoch [160/200], Train Loss: 1.4387
Epoch [180/200], Train Loss: 1.4286
Epoch [200/200], Train Loss: 1.3790
Time taken[cnn]: 27.58 seconds

> 标签传播:
构建SSPTM...
预测伪标签...
Time taken[diffusion]: 51.31 seconds

> Time taken[train 5]: 78.91 seconds

> 投票结果:
Accuracy[vote:5]: 0.9179
History accuracy: [0.9185, 0.9181, 0.918, 0.9182, 0.9179]
Total time taken: 394.29 seconds

======================== 第6轮训练 ========================
> cnn训练:
Epoch [20/200], Train Loss: 2.0260
Epoch [40/200], Train Loss: 1.8690
Epoch [60/200], Train Loss: 1.7538
Epoch [80/200], Train Loss: 1.6562
Epoch [100/200], Train Loss: 1.6050
Epoch [120/200], Train Loss: 1.5387
Epoch [140/200], Train Loss: 1.4888
Epoch [160/200], Train Loss: 1.4464
Epoch [180/200], Train Loss: 1.4105
Epoch [200/200], Train Loss: 1.3702
Time taken[cnn]: 27.56 seconds

> 标签传播:
构建SSPTM...
预测伪标签...
Time taken[diffusion]: 51.74 seconds

> Time taken[train 6]: 79.33 seconds

> 投票结果:
Accuracy[vote:6]: 0.9181
History accuracy: [0.9185, 0.9181, 0.918, 0.9182, 0.9179, 0.9181]
Total time taken: 473.62 seconds

======================== 第7轮训练 ========================
> cnn训练:
Epoch [20/200], Train Loss: 2.0059
Epoch [40/200], Train Loss: 1.8534
Epoch [60/200], Train Loss: 1.7194
Epoch [80/200], Train Loss: 1.6509
Epoch [100/200], Train Loss: 1.5972
Epoch [120/200], Train Loss: 1.5567
Epoch [140/200], Train Loss: 1.4965
Epoch [160/200], Train Loss: 1.4494
Epoch [180/200], Train Loss: 1.4247
Epoch [200/200], Train Loss: 1.4007
Time taken[cnn]: 26.00 seconds

> 标签传播:
构建SSPTM...
预测伪标签...
Time taken[diffusion]: 53.42 seconds

> Time taken[train 7]: 79.44 seconds

> 投票结果:
Accuracy[vote:7]: 0.9178
History accuracy: [0.9185, 0.9181, 0.918, 0.9182, 0.9179, 0.9181, 0.9178]
Total time taken: 553.05 seconds

======================== 第8轮训练 ========================
> cnn训练:
Epoch [20/200], Train Loss: 1.9975
Epoch [40/200], Train Loss: 1.8626
Epoch [60/200], Train Loss: 1.7569
Epoch [80/200], Train Loss: 1.6715
Epoch [100/200], Train Loss: 1.6095
Epoch [120/200], Train Loss: 1.5557
Epoch [140/200], Train Loss: 1.4921
Epoch [160/200], Train Loss: 1.4670
Epoch [180/200], Train Loss: 1.4196
Epoch [200/200], Train Loss: 1.3872
Time taken[cnn]: 26.51 seconds

> 标签传播:
构建SSPTM...
预测伪标签...
Time taken[diffusion]: 48.95 seconds

> Time taken[train 8]: 75.48 seconds

> 投票结果:
Accuracy[vote:8]: 0.9179
History accuracy: [0.9185, 0.9181, 0.918, 0.9182, 0.9179, 0.9181, 0.9178, 0.9179]
Total time taken: 628.53 seconds

======================== 第9轮训练 ========================
> cnn训练:
Epoch [20/200], Train Loss: 2.1212
Epoch [40/200], Train Loss: 1.8684
Epoch [60/200], Train Loss: 1.7448
Epoch [80/200], Train Loss: 1.6881
Epoch [100/200], Train Loss: 1.6089
Epoch [120/200], Train Loss: 1.5512
Epoch [140/200], Train Loss: 1.5002
Epoch [160/200], Train Loss: 1.4667
Epoch [180/200], Train Loss: 1.4282
Epoch [200/200], Train Loss: 1.3945
Time taken[cnn]: 25.93 seconds

> 标签传播:
构建SSPTM...
预测伪标签...
Time taken[diffusion]: 54.00 seconds

> Time taken[train 9]: 79.95 seconds

> 投票结果:
Accuracy[vote:9]: 0.9178
History accuracy: [0.9185, 0.9181, 0.918, 0.9182, 0.9179, 0.9181, 0.9178, 0.9179, 0.9178]
Total time taken: 708.49 seconds

======================== 第10轮训练 ========================
> cnn训练:
Epoch [20/200], Train Loss: 2.0298
Epoch [40/200], Train Loss: 1.8350
Epoch [60/200], Train Loss: 1.7255
Epoch [80/200], Train Loss: 1.6439
Epoch [100/200], Train Loss: 1.5692
Epoch [120/200], Train Loss: 1.5326
Epoch [140/200], Train Loss: 1.4780
Epoch [160/200], Train Loss: 1.4439
Epoch [180/200], Train Loss: 1.4113
Epoch [200/200], Train Loss: 1.3572
Time taken[cnn]: 26.00 seconds

> 标签传播:
构建SSPTM...
预测伪标签...
Time taken[diffusion]: 52.45 seconds

> Time taken[train 10]: 78.47 seconds

> 投票结果:
Accuracy[vote:10]: 0.9180
History accuracy: [0.9185, 0.9181, 0.918, 0.9182, 0.9179, 0.9181, 0.9178, 0.9179, 0.9178, 0.918]
Total time taken: 786.96 seconds

======================== 第11轮训练 ========================
> cnn训练:
Epoch [20/200], Train Loss: 1.9901
Epoch [40/200], Train Loss: 1.8181
Epoch [60/200], Train Loss: 1.7101
Epoch [80/200], Train Loss: 1.6150
Epoch [100/200], Train Loss: 1.5584
Epoch [120/200], Train Loss: 1.5064
Epoch [140/200], Train Loss: 1.4479
Epoch [160/200], Train Loss: 1.4185
Epoch [180/200], Train Loss: 1.3853
Epoch [200/200], Train Loss: 1.3453
Time taken[cnn]: 26.96 seconds

> 标签传播:
构建SSPTM...
预测伪标签...
Time taken[diffusion]: 52.95 seconds

> Time taken[train 11]: 79.93 seconds

> 投票结果:
Accuracy[vote:11]: 0.9179
History accuracy: [0.9185, 0.9181, 0.918, 0.9182, 0.9179, 0.9181, 0.9178, 0.9179, 0.9178, 0.918, 0.9179]
Total time taken: 866.89 seconds

======================== 第12轮训练 ========================
> cnn训练:
Epoch [20/200], Train Loss: 2.0677
Epoch [40/200], Train Loss: 1.8345
Epoch [60/200], Train Loss: 1.7255
Epoch [80/200], Train Loss: 1.6312
Epoch [100/200], Train Loss: 1.5754
Epoch [120/200], Train Loss: 1.5103
Epoch [140/200], Train Loss: 1.4616
Epoch [160/200], Train Loss: 1.4158
Epoch [180/200], Train Loss: 1.3900
Epoch [200/200], Train Loss: 1.3539
Time taken[cnn]: 26.35 seconds

> 标签传播:
构建SSPTM...
预测伪标签...
Time taken[diffusion]: 46.10 seconds

> Time taken[train 12]: 72.46 seconds

> 投票结果:
Accuracy[vote:12]: 0.9182
History accuracy: [0.9185, 0.9181, 0.918, 0.9182, 0.9179, 0.9181, 0.9178, 0.9179, 0.9178, 0.918, 0.9179, 0.9182]
Total time taken: 939.35 seconds

======================== 第13轮训练 ========================
> cnn训练:
Epoch [20/200], Train Loss: 2.0250
Epoch [40/200], Train Loss: 1.8365
Epoch [60/200], Train Loss: 1.7587
Epoch [80/200], Train Loss: 1.6571
Epoch [100/200], Train Loss: 1.6048
Epoch [120/200], Train Loss: 1.5512
Epoch [140/200], Train Loss: 1.5007
Epoch [160/200], Train Loss: 1.4483
Epoch [180/200], Train Loss: 1.4009
Epoch [200/200], Train Loss: 1.3800
Time taken[cnn]: 25.87 seconds

> 标签传播:
构建SSPTM...
预测伪标签...
Time taken[diffusion]: 48.86 seconds

> Time taken[train 13]: 74.75 seconds

> 投票结果:
Accuracy[vote:13]: 0.9180
History accuracy: [0.9185, 0.9181, 0.918, 0.9182, 0.9179, 0.9181, 0.9178, 0.9179, 0.9178, 0.918, 0.9179, 0.9182, 0.918]
Total time taken: 1014.10 seconds

======================== 第14轮训练 ========================
> cnn训练:
Epoch [20/200], Train Loss: 2.0479
Epoch [40/200], Train Loss: 1.8503
Epoch [60/200], Train Loss: 1.7192
Epoch [80/200], Train Loss: 1.6409
Epoch [100/200], Train Loss: 1.5859
Epoch [120/200], Train Loss: 1.5477
Epoch [140/200], Train Loss: 1.4963
Epoch [160/200], Train Loss: 1.4616
Epoch [180/200], Train Loss: 1.4123
Epoch [200/200], Train Loss: 1.3878
Time taken[cnn]: 26.13 seconds

> 标签传播:
构建SSPTM...
预测伪标签...
Time taken[diffusion]: 48.63 seconds

> Time taken[train 14]: 74.78 seconds

> 投票结果:
Accuracy[vote:14]: 0.9181
History accuracy: [0.9185, 0.9181, 0.918, 0.9182, 0.9179, 0.9181, 0.9178, 0.9179, 0.9178, 0.918, 0.9179, 0.9182, 0.918, 0.9181]
Total time taken: 1088.87 seconds

======================== 第15轮训练 ========================
> cnn训练:
Epoch [20/200], Train Loss: 2.0381
Epoch [40/200], Train Loss: 1.8258
Epoch [60/200], Train Loss: 1.7275
Epoch [80/200], Train Loss: 1.6495
Epoch [100/200], Train Loss: 1.5913
Epoch [120/200], Train Loss: 1.5301
Epoch [140/200], Train Loss: 1.4932
Epoch [160/200], Train Loss: 1.4473
Epoch [180/200], Train Loss: 1.4236
Epoch [200/200], Train Loss: 1.3753
Time taken[cnn]: 26.16 seconds

> 标签传播:
构建SSPTM...
预测伪标签...
Time taken[diffusion]: 48.06 seconds

> Time taken[train 15]: 74.23 seconds

> 投票结果:
Accuracy[vote:15]: 0.9180
History accuracy: [0.9185, 0.9181, 0.918, 0.9182, 0.9179, 0.9181, 0.9178, 0.9179, 0.9178, 0.918, 0.9179, 0.9182, 0.918, 0.9181, 0.918]
Total time taken: 1163.11 seconds

======================== 第16轮训练 ========================
> cnn训练:
Epoch [20/200], Train Loss: 2.0551
Epoch [40/200], Train Loss: 1.8430
Epoch [60/200], Train Loss: 1.7355
Epoch [80/200], Train Loss: 1.6439
Epoch [100/200], Train Loss: 1.5758
Epoch [120/200], Train Loss: 1.5201
Epoch [140/200], Train Loss: 1.4787
Epoch [160/200], Train Loss: 1.4505
Epoch [180/200], Train Loss: 1.3866
Epoch [200/200], Train Loss: 1.3597
Time taken[cnn]: 25.82 seconds

> 标签传播:
构建SSPTM...
预测伪标签...
Time taken[diffusion]: 46.37 seconds

> Time taken[train 16]: 72.20 seconds

> 投票结果:
Accuracy[vote:16]: 0.9180
History accuracy: [0.9185, 0.9181, 0.918, 0.9182, 0.9179, 0.9181, 0.9178, 0.9179, 0.9178, 0.918, 0.9179, 0.9182, 0.918, 0.9181, 0.918, 0.918]
Total time taken: 1235.31 seconds

======================== 第17轮训练 ========================
> cnn训练:
Epoch [20/200], Train Loss: 2.0656
Epoch [40/200], Train Loss: 1.8546
Epoch [60/200], Train Loss: 1.7131
Epoch [80/200], Train Loss: 1.6349
Epoch [100/200], Train Loss: 1.5595
Epoch [120/200], Train Loss: 1.5202
Epoch [140/200], Train Loss: 1.4723
Epoch [160/200], Train Loss: 1.4305
Epoch [180/200], Train Loss: 1.4000
Epoch [200/200], Train Loss: 1.3694
Time taken[cnn]: 27.11 seconds

> 标签传播:
构建SSPTM...
预测伪标签...
Time taken[diffusion]: 49.88 seconds

> Time taken[train 17]: 77.00 seconds

> 投票结果:
Accuracy[vote:17]: 0.9180
History accuracy: [0.9185, 0.9181, 0.918, 0.9182, 0.9179, 0.9181, 0.9178, 0.9179, 0.9178, 0.918, 0.9179, 0.9182, 0.918, 0.9181, 0.918, 0.918, 0.918]
Total time taken: 1312.31 seconds

======================== 第18轮训练 ========================
> cnn训练:
Epoch [20/200], Train Loss: 2.0183
Epoch [40/200], Train Loss: 1.8420
Epoch [60/200], Train Loss: 1.7230
Epoch [80/200], Train Loss: 1.6636
Epoch [100/200], Train Loss: 1.6105
Epoch [120/200], Train Loss: 1.5518
Epoch [140/200], Train Loss: 1.5066
Epoch [160/200], Train Loss: 1.4653
Epoch [180/200], Train Loss: 1.4229
Epoch [200/200], Train Loss: 1.3920
Time taken[cnn]: 25.79 seconds

> 标签传播:
构建SSPTM...
预测伪标签...
Time taken[diffusion]: 50.69 seconds

> Time taken[train 18]: 76.49 seconds

> 投票结果:
Accuracy[vote:18]: 0.9180
History accuracy: [0.9185, 0.9181, 0.918, 0.9182, 0.9179, 0.9181, 0.9178, 0.9179, 0.9178, 0.918, 0.9179, 0.9182, 0.918, 0.9181, 0.918, 0.918, 0.918, 0.918]
Total time taken: 1388.80 seconds

======================== 第19轮训练 ========================
> cnn训练:
Epoch [20/200], Train Loss: 2.0636
Epoch [40/200], Train Loss: 1.8402
Epoch [60/200], Train Loss: 1.7315
Epoch [80/200], Train Loss: 1.6660
Epoch [100/200], Train Loss: 1.5909
Epoch [120/200], Train Loss: 1.5352
Epoch [140/200], Train Loss: 1.4920
Epoch [160/200], Train Loss: 1.4327
Epoch [180/200], Train Loss: 1.3963
Epoch [200/200], Train Loss: 1.3651
Time taken[cnn]: 26.55 seconds

> 标签传播:
构建SSPTM...
预测伪标签...
Time taken[diffusion]: 51.78 seconds

> Time taken[train 19]: 78.35 seconds

> 投票结果:
Accuracy[vote:19]: 0.9180
History accuracy: [0.9185, 0.9181, 0.918, 0.9182, 0.9179, 0.9181, 0.9178, 0.9179, 0.9178, 0.918, 0.9179, 0.9182, 0.918, 0.9181, 0.918, 0.918, 0.918, 0.918, 0.918]
Total time taken: 1467.15 seconds

======================== 第20轮训练 ========================
> cnn训练:
Epoch [20/200], Train Loss: 2.0099
Epoch [40/200], Train Loss: 1.8133
Epoch [60/200], Train Loss: 1.7126
Epoch [80/200], Train Loss: 1.6453
Epoch [100/200], Train Loss: 1.5964
Epoch [120/200], Train Loss: 1.5520
Epoch [140/200], Train Loss: 1.4948
Epoch [160/200], Train Loss: 1.4601
Epoch [180/200], Train Loss: 1.4152
Epoch [200/200], Train Loss: 1.3882
Time taken[cnn]: 25.83 seconds

> 标签传播:
构建SSPTM...
预测伪标签...
Time taken[diffusion]: 48.44 seconds

> Time taken[train 20]: 74.28 seconds

> 投票结果:
Accuracy[vote:20]: 0.9180
History accuracy: [0.9185, 0.9181, 0.918, 0.9182, 0.9179, 0.9181, 0.9178, 0.9179, 0.9178, 0.918, 0.9179, 0.9182, 0.918, 0.9181, 0.918, 0.918, 0.918, 0.918, 0.918, 0.918]
Total time taken: 1541.43 seconds

======================== 第21轮训练 ========================
> cnn训练:
Epoch [20/200], Train Loss: 2.1208
Epoch [40/200], Train Loss: 1.8494
Epoch [60/200], Train Loss: 1.7388
Epoch [80/200], Train Loss: 1.6604
Epoch [100/200], Train Loss: 1.5946
Epoch [120/200], Train Loss: 1.5361
Epoch [140/200], Train Loss: 1.4941
Epoch [160/200], Train Loss: 1.4537
Epoch [180/200], Train Loss: 1.4145
Epoch [200/200], Train Loss: 1.3730
Time taken[cnn]: 25.79 seconds

> 标签传播:
构建SSPTM...
预测伪标签...
Time taken[diffusion]: 50.64 seconds

> Time taken[train 21]: 76.46 seconds

> 投票结果:
Accuracy[vote:21]: 0.9179
History accuracy: [0.9185, 0.9181, 0.918, 0.9182, 0.9179, 0.9181, 0.9178, 0.9179, 0.9178, 0.918, 0.9179, 0.9182, 0.918, 0.9181, 0.918, 0.918, 0.918, 0.918, 0.918, 0.918, 0.9179]
Total time taken: 1617.89 seconds

======================== 第22轮训练 ========================
> cnn训练:
Epoch [20/200], Train Loss: 1.9943
Epoch [40/200], Train Loss: 1.8189
Epoch [60/200], Train Loss: 1.7073
Epoch [80/200], Train Loss: 1.6392
Epoch [100/200], Train Loss: 1.5720
Epoch [120/200], Train Loss: 1.5273
Epoch [140/200], Train Loss: 1.4864
Epoch [160/200], Train Loss: 1.4476
Epoch [180/200], Train Loss: 1.4215
Epoch [200/200], Train Loss: 1.3817
Time taken[cnn]: 25.90 seconds

> 标签传播:
构建SSPTM...
预测伪标签...
Time taken[diffusion]: 50.80 seconds

> Time taken[train 22]: 76.71 seconds

> 投票结果:
Accuracy[vote:22]: 0.9181
History accuracy: [0.9185, 0.9181, 0.918, 0.9182, 0.9179, 0.9181, 0.9178, 0.9179, 0.9178, 0.918, 0.9179, 0.9182, 0.918, 0.9181, 0.918, 0.918, 0.918, 0.918, 0.918, 0.918, 0.9179, 0.9181]
Total time taken: 1694.60 seconds

======================== 第23轮训练 ========================
> cnn训练:
Epoch [20/200], Train Loss: 2.0040
Epoch [40/200], Train Loss: 1.8218
Epoch [60/200], Train Loss: 1.7324
Epoch [80/200], Train Loss: 1.6414
Epoch [100/200], Train Loss: 1.5750
Epoch [120/200], Train Loss: 1.5283
Epoch [140/200], Train Loss: 1.4704
Epoch [160/200], Train Loss: 1.4295
Epoch [180/200], Train Loss: 1.3835
Epoch [200/200], Train Loss: 1.3606
Time taken[cnn]: 26.18 seconds

> 标签传播:
构建SSPTM...
预测伪标签...
Time taken[diffusion]: 55.58 seconds

> Time taken[train 23]: 81.78 seconds

> 投票结果:
Accuracy[vote:23]: 0.9180
History accuracy: [0.9185, 0.9181, 0.918, 0.9182, 0.9179, 0.9181, 0.9178, 0.9179, 0.9178, 0.918, 0.9179, 0.9182, 0.918, 0.9181, 0.918, 0.918, 0.918, 0.918, 0.918, 0.918, 0.9179, 0.9181, 0.918]
Total time taken: 1776.38 seconds

======================== 第24轮训练 ========================
> cnn训练:
Epoch [20/200], Train Loss: 2.0341
Epoch [40/200], Train Loss: 1.8503
Epoch [60/200], Train Loss: 1.7264
Epoch [80/200], Train Loss: 1.6441
Epoch [100/200], Train Loss: 1.5550
Epoch [120/200], Train Loss: 1.5048
Epoch [140/200], Train Loss: 1.4487
Epoch [160/200], Train Loss: 1.4112
Epoch [180/200], Train Loss: 1.3846
Epoch [200/200], Train Loss: 1.3431
Time taken[cnn]: 26.92 seconds

> 标签传播:
构建SSPTM...
预测伪标签...
Time taken[diffusion]: 50.18 seconds

> Time taken[train 24]: 77.13 seconds

> 投票结果:
Accuracy[vote:24]: 0.9180
History accuracy: [0.9185, 0.9181, 0.918, 0.9182, 0.9179, 0.9181, 0.9178, 0.9179, 0.9178, 0.918, 0.9179, 0.9182, 0.918, 0.9181, 0.918, 0.918, 0.918, 0.918, 0.918, 0.918, 0.9179, 0.9181, 0.918, 0.918]
Total time taken: 1853.51 seconds

======================== 第25轮训练 ========================
> cnn训练:
Epoch [20/200], Train Loss: 2.0504
Epoch [40/200], Train Loss: 1.8537
Epoch [60/200], Train Loss: 1.7341
Epoch [80/200], Train Loss: 1.6625
Epoch [100/200], Train Loss: 1.5951
Epoch [120/200], Train Loss: 1.5354
Epoch [140/200], Train Loss: 1.4865
Epoch [160/200], Train Loss: 1.4383
Epoch [180/200], Train Loss: 1.4001
Epoch [200/200], Train Loss: 1.3671
Time taken[cnn]: 26.18 seconds

> 标签传播:
构建SSPTM...
预测伪标签...
Time taken[diffusion]: 53.70 seconds

> Time taken[train 25]: 79.91 seconds

> 投票结果:
Accuracy[vote:25]: 0.9179
History accuracy: [0.9185, 0.9181, 0.918, 0.9182, 0.9179, 0.9181, 0.9178, 0.9179, 0.9178, 0.918, 0.9179, 0.9182, 0.918, 0.9181, 0.918, 0.918, 0.918, 0.918, 0.918, 0.918, 0.9179, 0.9181, 0.918, 0.918, 0.9179]
Total time taken: 1933.42 seconds