# -*- coding: utf-8 -*-
# @Time    : 2021/7/16 15:01
# @Author  : yuting
# @FileName: timeData_WRFmode.py
import pandas as pd
import os
from loguru import logger

# WRF文件路径
SIMULATED_FILE = "./模拟数据"
# 提取到的做时间序列预测的模拟数据
OUTPUT_FILE = "./时序数据"

# 选取的地理位置
NX = 1
NY = 1
NL = 1


# 提取某一点三个月的时间数据
def getTimeData(PATH):
    data = pd.read_csv(PATH)
    data_choose = data.loc[(data['nx'] == NX) & (data['ny'] == NY) & (data['nl'] == NL)]
    return data_choose


if __name__ == "__main__":
    g = os.walk(SIMULATED_FILE)
    for path, dir_list, file_list in g:
        for file_name in file_list:
            data = getTimeData(os.path.join(path, file_name))

            # 处理异常数据，2月4日的数据只有十条
            if file_name == "wrfout_d02_2017-02-04_00_00_00.csv":
                # 异常数据，打算不输出，若想输出，则执行以下代码
                '''
                data.to_csv(OUTPUT_FILE + "\\nx=" + str(NX) + "ny=" + str(NY) + "nl=" + str(NL) + "_wrfMode.csv", mode='a',
                            encoding="utf-8",
                            header=None, index=None)
                '''
                continue
            elif file_name == "wrfout_d02_2017-01-01_00_00_00.csv":
                # data.drop([len(data) - 1], inplace=True)
                data.to_csv(OUTPUT_FILE + "\\nx=" + str(NX) + "ny=" + str(NY) + "nl=" + str(NL) + "_wrfMode.csv",
                            encoding="utf-8",
                            header=True, index=None)
            else:
                # data.drop([len(data) - 1], inplace=True)
                data.to_csv(OUTPUT_FILE + "\\nx=" + str(NX) + "ny=" + str(NY) + "nl=" + str(NL) + "_wrfMode.csv",
                            mode='a',
                            encoding="utf-8",
                            header=None, index=None)

            logger.info("已加入" + file_name[11:21] + "的数据")
