from netCDF4 import Dataset
import numpy as np
import pandas as pd
import itertools
import os

# 原始文件路径，从硬盘读入
RAW_DATA = "F:\wrf_out\\2017"
# 输出文件路径
OUTPUT_DATA = "./模拟数据\\"


def readWRFData(PATH):
    path = PATH
    raw_data = Dataset(path)

    # 所有使用到的变量名
    used_features = [
        'Times',  # 时间
        'XLAT',  # 维度
        'XLONG',  # 经度
        'U',  # 东西风力
        'V',  # 南北风力
        'PH',  # perturbation geopotential，扰动位势
        'PHB',  # base-state geopotential，基态位势
        'T',  # perturbation potential temperature，扰动潜在温度
        'P',  # perturbation pressure，扰动压强
        'PB',  # base state pressure，基态压强
        'HGT',  # Terrain height，海拔高度
        'QVAPOR',  # 水汽混合比
    ]

    raw_data.variables['T'][:]  # 访问变量数据的方法，返回numpy掩码数组
    np.array(raw_data.variables['T'][:])  # 可以直接转成numpy数组
    # 利用‘P’和‘PB’计算气压值pressure
    pressure = (raw_data.variables['P'][:] + raw_data.variables['PB'][:]) / 100
    print("气压矩阵维度：")
    print(pressure.shape)

    NT = pressure.shape[0]
    NL = pressure.shape[1]
    NY = pressure.shape[2]
    NX = pressure.shape[3]

    # (25,85,93,90)(nt,nl,ny,nx)第一维标签
    nx = pd.Series(np.arange(1, NX + 1))  # nx东西格点数
    ny = pd.Series(np.arange(1, NY + 1))  # ny南北格点数
    nl = pd.Series(np.arange(1, NL + 1))
    nt = pd.Series(np.arange(1, NT + 1))
    # 展开标签
    # index的笛卡尔乘积。注意：高维在前，低维在后
    prod = itertools.product(nx, ny, nl, nt)
    # 转换为DataFrame
    prod = pd.DataFrame([x for x in prod])
    prod.columns = ['nx', 'ny', 'nl', 'nt']

    # 利用‘PH’和‘PHB’计算高度值height
    PH, PHB = raw_data.variables['PH'][:], raw_data.variables['PHB'][:]
    PHH = 0.5 * (PH[:, :85, :, :] + PH[:, 1:86, :, :])
    PHBH = 0.5 * (PHB[:, :85, :, :] + PHB[:, 1:86, :, :])
    GEOPT = PHH + PHBH
    height = GEOPT / 9.81  # 求高度值
    print("高度矩阵维度:")
    print(height.shape)

    # 利用‘T’和‘pressure’计算温度值T
    T = raw_data.variables['T'][:]
    Tem = (T + 300.) / (1000. / pressure) ** 0.286
    print("温度矩阵维度:")
    print(Tem.shape)

    # 利用‘QVAPOR’计算水气压值
    QVAPOR = raw_data.variables['QVAPOR'][:]  # 水汽混合比
    e = QVAPOR * pressure / (0.622 + QVAPOR)

    XLONG = raw_data.variables['XLONG'][:]  # 读取经度值
    HGT = raw_data.variables['HGT'][:]  # 读取海拔高度
    XLAT = raw_data.variables['XLAT'][:]  # 读取纬度值

    xlong, xlat, hgt = np.zeros((NT, 93, 90, 85)), np.zeros((NT, 93, 90, 85)), np.zeros((NT, 93, 90, 85))
    # 转化为四维数组
    for i in range(85):
        xlong[:, :, :, i] = XLONG[:, :, :]
        xlat[:, :, :, i] = XLAT[:, :, :]
        hgt[:, :, :, i] = HGT[:, :, :]

    # 根据Fortran的代码转化数据维度  (nx东西格点数, ny南北格点数, nl垂直层数, nt时次数)
    e = e.transpose((3, 2, 1, 0))
    Tem = Tem.transpose((3, 2, 1, 0))
    QVAPOR = QVAPOR.transpose((3, 2, 1, 0))
    pressure = pressure.transpose((3, 2, 1, 0))
    height = height.transpose((3, 2, 1, 0))
    xlong = xlong.transpose((2, 1, 3, 0))
    hgt = hgt.transpose((2, 1, 3, 0))
    xlat = xlat.transpose((2, 1, 3, 0))
    z = height - hgt

    # 计算折射率，NN是原始折射率，MM是修正折射率
    re = 6371.
    NN = (3.73 * 10 ** 5 * e) / (Tem ** 2) + (77.6 * pressure) / (Tem)
    MM = (3.73 * 10 ** 5 * e) / (Tem ** 2) + (77.6 * pressure) / (Tem) + ((height - hgt) / (re * 1000.) * 10 ** 6)

    a = xlong.flatten()
    a = pd.Series(a)
    a.name = 'xlong'

    b = xlat.flatten()
    b = pd.Series(b)
    b.name = 'xlat'

    c = hgt.flatten()
    c = pd.Series(c)
    c.name = 'hgt'

    o = height.flatten()
    o = pd.Series(o)
    o.name = 'height'

    d = z.flatten()
    d = pd.Series(d)
    d.name = 'z'

    f = e.flatten()
    f = pd.Series(f)
    f.name = 'e'

    g = pressure.flatten()
    g = pd.Series(g)
    g.name = 'pressure'

    h = Tem.flatten()
    h = pd.Series(h)
    h.name = 'Tem'

    l = QVAPOR.flatten()
    l = pd.Series(l)
    l.name = 'QVAPOR'

    mm = MM.flatten()
    mm = pd.Series(mm)
    mm.name = 'MM'

    nn = NN.flatten()
    nn = pd.Series(nn)
    nn.name = 'NN'

    # 最终数据，合并成一个DataFrame
    data = pd.concat([prod, a, b, c, o, d, f, g, h, l, mm, nn], axis=1)
    data1 = pd.DataFrame(data)
    data2 = data1.iloc[0:NT * NL * NY, :]
    return data2


if __name__ == '__main__':

    g = os.walk(RAW_DATA)

    for path, dir_list, file_list in g:
        for file_name in file_list:
            print(os.path.join(path, file_name))
            data2 = readWRFData(os.path.join(path, file_name))
            data2.to_csv(OUTPUT_DATA + file_name + ".csv", encoding="utf-8", header=True, index=None)
