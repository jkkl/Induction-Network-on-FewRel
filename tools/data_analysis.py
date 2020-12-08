# !/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: yuanzhuo@xiaoice.com
Created on: 2020/11/3 10:39
'''

import pandas as pd


class DataAnalysis:
    '''
    数据处理、统计、观察
    '''

    def __init__(self, input_file):
        '''
        input_file: cat_des\tquery
        '''
        self.data_dir = input_file[0: input_file.rfind("\\")]
        self.input_file_name = input_file[input_file.rfind("\\") + 1: input_file.rfind(".")]
        self.input_file = input_file
        self.input_pd = self.load_data(input_file)
        # 将输入文件的类别信息写入到excel
        self.class_static()

    def load_data(self, input_file):
        '''
        使用pandas加载数据
        '''
        try:
            input_pd = pd.read_excel(input_file)
        except Exception:
            input_pd = pd.read_table(input_file)

        return input_pd

    def class_static(self):
        '''
        统计输入文件的统计特征
        '''
        class_num = self.input_pd.iloc[:, 0].value_counts()
        output_path = self.data_dir + "\\" + self.input_file_name + "_class_desc.xlsx"
        class_desc_df = class_num.to_frame()
        class_desc_df.columns = ['count']
        class_desc_static = class_num.describe()
        class_desc_static.columns = ['static']
        with pd.ExcelWriter(output_path) as writer:
            class_desc_df.to_excel(writer, sheet_name='class_desc_distribute')
            class_desc_static.to_excel(writer, sheet_name='class_desc_static')
        return class_num



if __name__ == '__main__':
    train_file = "C:\\Users\\User\\Documents\\project\\intention\\data\\General_Intention_134_v1_new.xlsx"
    da = DataAnalysis(train_file)