# !/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: yuanzhuo@xiaoice.com
Created on: 2020/11/3 16:52
'''
import json

import pandas as pd

from tools.common import load_data2df


#定义数据列名
EMOTION_CATEGORY = "EmotionCategory"
INTENT_DESC = "IntentDesc"
REQUEST_CONTENT = "RequestContent"
ADDITIONAL_MESSAGE = "AdditionalMessage"


class DataProcess:
    '''
    处理日志文件，将日志、标注文件转为训练数据
    '''



    @staticmethod
    def cat_info(cat_info_file):
        cat_info = load_data2df(cat_info_file)
        return cat_info

    @staticmethod
    def get_sample_cat_info(label_file):
        sample_data = DataProcess.get_sample_from_label_file(label_file)
        cat_count_df = sample_data["IntentDesc"].value_counts().reset_index(name="count")
        cat_count_df.rename(columns={'index': 'IntentDesc'}, inplace=True)
        # 获取测评数据中样本书大于10的类别
        label_cat_series = cat_count_df[cat_count_df["count"] > 10]["IntentDesc"]
        return label_cat_series

    @staticmethod
    def cat_diff(cat_info_file, label_file):
        '''
        找出类别不同的数据：cat_info_file - label_file
        '''
        cat_info = DataProcess.cat_info(cat_info_file)
        label_cat_series = DataProcess.get_sample_cat_info(label_file)

        # 求两个series的差集
        diff = DataProcess.get_diff_series(cat_info["Intention"], label_cat_series)
        diff.to_csv("./cat_not_in_label.txt", sep='\t', index=False, header=False)

        print(diff)
        return diff


    @staticmethod
    def get_diff_series(A, B):
        diff = pd.Series(list(set(A).difference(set(B))))
        return diff



    @staticmethod
    def get_sample_from_label_file(label_file):
        '''
        将标注数据转为样本数据
        '''
        label_data = load_data2df(label_file)
        true_data = label_data[label_data['Label'] == 1]
        sample_data = true_data.apply(DataProcess.parse_row, axis=1)
        return sample_data


    @staticmethod
    def trans_label2sample(label_file, sample_file, off_size):
        sample_data = DataProcess.get_sample_from_label_file(label_file)
        class_num = sample_data.iloc[:, 1].value_counts()
        class_desc_static = class_num.describe()
        class_desc_static.columns = ['static']
        with pd.ExcelWriter(sample_file) as writer:
            sample_data.to_excel(writer, sheet_name='sample')
            class_desc_static.to_excel(writer, sheet_name='static')

    @staticmethod
    def parse_row(row):
        '''
        解析输入数据的一行，按照列名去除
        融合emotion和intention
        '''
        request_content = row[REQUEST_CONTENT]
        json_row = json.loads(row[ADDITIONAL_MESSAGE])
        # 合并emotion与intention数据
        intent_desc = "none"
        if INTENT_DESC in json_row:
            intent_desc = json_row[INTENT_DESC]
        elif EMOTION_CATEGORY in json_row:
            intent_desc = json_row[EMOTION_CATEGORY]
        return pd.Series({REQUEST_CONTENT: request_content, INTENT_DESC: intent_desc})

    @staticmethod
    def data_filter(data_frame):
        # filter_cat = ['晚睡/睡不好']
        data_frame_filter = data_frame[~ (data_frame["IntentDesc"] == '晚睡/睡不好')]
        data_frame_filter = data_frame_filter.dropna()
        return data_frame_filter


def test_data_describe(test_label_data, emotion_intention_map_file=None, intention_134_cat_dim=None):
    '''
    统计分析测试文件中的类别情况
    '''
    #加载标注文件
    test_df = DataProcess.get_sample_from_label_file(test_label_data)
    test_df = DataProcess.data_filter(test_df)
    # emotion_intention_map = load_data2df(emotion_intention_map_file)
    # emotion_intention_dict = emotion_intention_map.set_index(EMOTION_CATEGORY)[INTENT_DESC].to_dict()

    # test_df[INTENT_DESC].replace(emotion_intention_dict, inplace=True)
    test_cat = test_df[INTENT_DESC].drop_duplicates()
    cat_dim = load_data2df(intention_134_cat_dim)
    # 校验数据是否有问题
    cat_not_in_test = DataProcess.get_diff_series(cat_dim[INTENT_DESC], test_cat)
    cat_not_in_dim = DataProcess.get_diff_series(test_cat, cat_dim[INTENT_DESC])
    cat_not_in_test.to_csv('cat_not_int_test.csv', index=False)
    print(f'cat not in test:!!! Please add same sample to test data\n{cat_not_in_test.to_string()}')
    if cat_not_in_dim.count() > 0:
        print(f'cat not in dim：！！！check test label data\n{cat_not_in_dim.to_string()}')

    assert cat_not_in_dim.count() == 0



    #统计各个类别分布



    #找出小于10个和没有出现的类别


    #分别输出全集和96测评集

    pass


if __name__ == '__main__':
    # data_dir = "C:\\Users\\User\\Documents\\project\\intention\\data\\"
    # label_file = data_dir + "IntentionDetection_汇总.xlsx"
    # cat_info_file = data_dir + "queryintention.response.leilei.txt"
    # # DataProcess.trans_label2sample(label_file, label_file+".sample.xlsx", 1500)
    # DataProcess.cat_diff(cat_info_file, label_file)
    test_data_dir ="C:\\Users\\User\\Documents\\project\\intention\\data\\test\\"
    test_label_data = test_data_dir + "intention_test_labeled_data.xlsx"
    emotion_intention_map_file = test_data_dir + "emotion_intention_map.txt"
    # intention_cat_dim_file = test_data_dir + "intention_134_cat_dim.txt"
    intention_cat_dim_file = test_data_dir + "intention_134_cat_dim_online.txt"
    test_data_describe(test_label_data, emotion_intention_map_file, intention_cat_dim_file)