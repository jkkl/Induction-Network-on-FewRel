# !/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: yuanzhuo@xiaoice.com
Created on: 2020/11/5 20:50
'''
import pandas as pd

'''
公共工具文件
'''


def load_data2df(input_file):
    '''
    使用pandas加载数据
    '''
    try:
        input_pd = pd.read_excel(input_file)
    except Exception:
        input_pd = pd.read_table(input_file)

    return input_pd