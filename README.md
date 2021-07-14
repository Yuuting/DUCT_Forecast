# Duct data time series forecast

本次数据处理的数据集来自南海区域2017年1月至三月的气象数据，其中各项气象参数所代表的含义如下表所示。
nx|ny|nl|nt|xlong|xlat|hgt|height
--|--|--|--|--|--|--
东西格点数|南北格点数|层数|时序数|经度|纬度|高度|海拔高度

z|e|pressure|Tem|QVAPOR|MM|NN
--|--|--|--|--|--
实际高度|水汽压|气压|温度|水汽混合比|大气折射率|修正大气折射率

由于原始数据使用气象专用格式存储，我这里已使用`getWRFData.py`程序提取到了1-3月nx=1的时序数据，你可以在[这里](https://pan.baidu.com/s/1KXLhmSZLxvm5wBlsjbqXHg 
)（提取码：`sk23`）下载。

## usage
```bash
python timeData.py
```

> 提取`nx=1`、`ny=1`、`nl=1`的时序数据，当然也可更改提取数据的位置坐标。将提取到的数据放入`时序数据`文件夹下。

```bash
python timeSeries.py
```

> 使用LSTM训练时序数据

- 数据预处理部分可以观察原始数据波动情况。

- 使用75%的数据作为训练集，25%数据作为测试集，训练效果如图`lstm_reg.png`所示。

- LSTM网络参数为：

  ```
  inp_dim = 5
  out_dim = 4
  mid_dim = 8
  mid_layers = 1
  batch_size = 48
  ```

