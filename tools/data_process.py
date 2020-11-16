# !/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: yuanzhuo@xiaoice.com
Created on: 2020/11/3 16:52
'''
import json

import pandas as pd

from tools.common import *


#定义数据列名
EMOTION_CATEGORY = "EmotionCategory"
INTENT_DESC = "IntentDesc"
REQUEST_CONTENT = "RequestContent"
ADDITIONAL_MESSAGE = "AddtionalMessage"
LABEL = "Label"
INTENT_DESC_96 = "IntentDesc_96"
EMOTION_DESC = "EmotionDesc"


#INPUT/OUTPUT
OUTPUT_SUFFIX = ".analysis.xlsx"
FOR_LABEL_SUFFIX = "_for_label.xlsx"
LABEL_DATA_SOURCE_SUFFIX = ".label_data_source.xlsx"
NEG_LABEL_DATA_SOURCE_SUFFIX = ".neg_label_data_source.xlsx"
SAMPLE_SUFFIX = '.sample.xlsx'
CAT_NOT_IN_DATA_SUFFIX = '.lackcat.xlsx'
NEG_CAT_NOT_IN_DATA_SUFFIX = '.neglackcat.xlsx'
TEST_FILE_PREFIX = 'test_sample'

#sheet name
LACK_CAT_SHEET="lack_sample_cat"


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
    def get_diff_cat_from_file(sample_file_path, cat_dim_file_path, sheet_name_of_dim_file=None, out_file=None):
        log_data = DataProcess.get_sample_from_sample_file(sample_file_path)
        cat_dim = load_data2df(cat_dim_file_path, sheet_name=sheet_name_of_dim_file)
        out_file = get_file_path_without_suffix(sample_file_path) + CAT_NOT_IN_DATA_SUFFIX
        DataProcess.get_diff_cat(log_data, cat_dim, out_file)


    @staticmethod
    def get_diff_cat(data_source, cat_dim, out_file, min_count=10):
        # 样本数大于十
        data_cat = data_source[INTENT_DESC].value_counts().reset_index(name="count")
        data_cat.rename(columns={'index': 'IntentDesc'}, inplace=True)
        data_cat = data_cat[data_cat["count"] > min_count]
        test_cat = data_cat[INTENT_DESC]
        cat_not_in_dim = pd.Series(list(set(test_cat).difference(set(cat_dim[INTENT_DESC]))))
        cat_not_in_data = pd.Series(list(set(cat_dim[INTENT_DESC]).difference(set(test_cat))))
        if cat_not_in_dim.count() > 0:
            print(f'Warning:!!!some cat not in dim {cat_not_in_dim.to_string()}')

        if cat_not_in_data.count() == 0:
            print(f'Message: every cat has data')
            return None
        else:
            print(f'Warning: {cat_not_in_data.count()} cats no data!!!')
            cat_not_in_data = cat_not_in_data.to_frame()
            cat_not_in_data.columns = [INTENT_DESC]
            print(cat_not_in_data.to_string())
            with pd.ExcelWriter(out_file) as writer:
                cat_not_in_data.to_excel(writer, index=False)
            return cat_not_in_data



    @staticmethod
    def get_sample_from_label_file(label_file, label=1):
        '''
        从标注了正误的标注集中读取数据：
        label： 数据过滤标识 1正例、0负例,如果没有label列，默认都是正例
        '''
        label_data = load_data2df(label_file)
        true_data = label_data
        # 如果标注集中没有label字段默认都是true data
        if "Label" in label_data.columns:
            true_data = label_data[label_data['Label'] == label]
        sample_data = true_data.apply(DataProcess.parse_row, axis=1)
        return sample_data

    @staticmethod
    def get_sample_from_log_file(log_file):
        '''
        读入原始日志数据,输出采样数据需要人工标注
        '''
        log_data = load_data2df(log_file)
        # ResquestContent AddtionalMessage and mix emotion and intention cat
        sample_data = log_data.apply(DataProcess.parse_row, axis=1)
        return sample_data

    @staticmethod
    def get_sample_from_sample_file(sample_file):
        '''
        读入直接打标类别的数据集
        '''
        sample_data = load_data2df(sample_file)
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
        # 合并emotion与intention数据
        intent_desc = "none"
        if ADDITIONAL_MESSAGE in row:
            #读取原始日志数据
            try:
                json_row = json.loads(row[ADDITIONAL_MESSAGE])
                if INTENT_DESC in json_row:
                    intent_desc = json_row[INTENT_DESC]
                elif EMOTION_CATEGORY in json_row:
                    intent_desc = json_row[EMOTION_CATEGORY]
            except Exception:
                print(f'bad line: {row}')
                return
        elif INTENT_DESC in row:
            # 读取标注好类别的数据
            intent_desc = row[INTENT_DESC]
        return pd.Series({REQUEST_CONTENT: request_content, INTENT_DESC: intent_desc})

    @staticmethod
    def data_filter(data_frame):
        # filter_cat = ['晚睡/睡不好']
        data_frame_filter = data_frame[~ (data_frame["IntentDesc"] == '晚睡/睡不好')]
        data_frame_filter = data_frame_filter.dropna()
        return data_frame_filter


def test_data_describe(test_label_data, emotion_intention_map_file=None, intention_134_cat_dim=None):
    '''
    统计分析测试文件中的类别情况,将统计信息输出到文件
    '''
    writer = pd.ExcelWriter(test_label_data + OUTPUT_SUFFIX)
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
    # TODO: rename cat column to IntentDesc
    cat_not_in_dim = DataProcess.get_diff_series(test_cat, cat_dim[INTENT_DESC])
    cat_not_in_test.to_csv('cat_not_int_test.csv', index=False)
    print(f'cat not in test:!!! Please add same sample to test data\n{cat_not_in_test.to_string()}')
    if cat_not_in_dim.count() > 0:
        print(f'cat not in dim：！！！check test label data\n{cat_not_in_dim.to_string()}')

    assert cat_not_in_dim.count() == 0

    #统计各个类别分布
    test_cat_count = test_df[INTENT_DESC].value_counts()
    test_cat_describe = test_cat_count.describe()
    test_cat_count.to_excel(writer, sheet_name='count')
    test_cat_describe.to_excel(writer, sheet_name='desc')
    cat_not_in_test.to_excel(writer, sheet_name='lack_sample_cat')
    cat_not_in_dim.to_excel(writer, sheet_name='error_cat')
    writer.save()


    #找出小于10个和没有出现的类别


    #分别输出全集和96测评集

    pass


def sample_data_from_log(log_file_path, max_per_cat, cat_dim_file, sheet_name_of_dim_file=0, out_file_path=None):
    '''
    从原始日志中采样需要标注的类别样本
    '''
    # 加载log数据
    log_data = DataProcess.get_sample_from_log_file(log_file_path)
    # 加载dim表
    cat_dim = load_data2df(cat_dim_file, sheet_name=sheet_name_of_dim_file)

    # 过滤出cat_dim中的样本
    # log_data = log_data[~log_data[INTENT_DESC].isin(cat_dim[INTENT_DESC])]
    log_data = log_data[log_data[INTENT_DESC].isin(cat_dim[INTENT_DESC])]

    # 分组采样，取head n
    sample_data = log_data.groupby([INTENT_DESC]).head(max_per_cat)

    # 保存数据
    log_data_sample_file = out_file_path
    if not log_data_sample_file:
        log_data_sample_file = get_file_path_without_suffix(log_file_path) + SAMPLE_SUFFIX
    with pd.ExcelWriter(log_data_sample_file) as writer:
        sample_data.to_excel(writer, sheet_name=f'sample_log_for_annotation_max_{max_per_cat}', index=False)



def merge_label_source_data(label_data_file_list, cat_dim_file, cat_sheet_name=None):
    '''
    加载标注数据集，找出其中的缺失类别数据
    return label_data, lack_cat_dim
    '''
    # 加载标注数据
    label_data_list = []
    for label_data_file in label_data_file_list:
        label_data_list.append(DataProcess.get_sample_from_label_file(label_data_file))
    label_data = pd.concat(label_data_list)
    cat_dim = load_data2df(cat_dim_file, sheet_name=cat_sheet_name)
    label_data = label_data[label_data[INTENT_DESC].isin(cat_dim[INTENT_DESC])]

    # 保存数据
    label_data_source_file = get_file_path_without_suffix(label_data_file_list[0]) + LABEL_DATA_SOURCE_SUFFIX
    with pd.ExcelWriter(label_data_source_file) as writer:
        label_data.to_excel(writer, index=False)

    # 输出缺失数据类别
    lack_cat_dim_file = get_file_path_without_suffix(label_data_source_file) + CAT_NOT_IN_DATA_SUFFIX
    DataProcess.get_diff_cat(label_data, cat_dim, lack_cat_dim_file)


def merge_pos_label_source_data(label_data_file_list, cat_dim_file, cat_sheet_name=None, min_count=10):
    '''
    加载标注数据集，找出其中的缺失类别数据
    return label_data, lack_cat_dim
    '''
    # 加载标注数据
    label_data_list = []
    for label_data_file in label_data_file_list:
        label_data_list.append(DataProcess.get_sample_from_label_file(label_data_file))
    label_data = pd.concat(label_data_list)
    cat_dim = load_data2df(cat_dim_file, sheet_name=cat_sheet_name)
    label_data = label_data[label_data[INTENT_DESC].isin(cat_dim[INTENT_DESC])]
    label_data[LABEL] = 1

    # 保存数据
    label_data_source_file = get_file_path_without_suffix(label_data_file_list[0]) + LABEL_DATA_SOURCE_SUFFIX
    with pd.ExcelWriter(label_data_source_file) as writer:
        label_data.to_excel(writer, index=False)

    # 输出缺失数据类别
    lack_cat_dim_file = get_file_path_without_suffix(label_data_source_file) + CAT_NOT_IN_DATA_SUFFIX
    cat_not_in_data = DataProcess.get_diff_cat(label_data, cat_dim, lack_cat_dim_file, min_count=min_count)

    return cat_not_in_data



def merge_neg_label_source_data(label_data_file_list, cat_dim_file, cat_sheet_name=None):
    '''
    加载标注数据集，找出其中的缺失类别数据
    return label_data, lack_cat_dim
    '''
    # 加载标注数据
    label_data_list = []
    for label_data_file in label_data_file_list:
        label_data_list.append(DataProcess.get_sample_from_label_file(label_data_file, label=0))
    label_data = pd.concat(label_data_list)
    cat_dim = load_data2df(cat_dim_file, sheet_name=cat_sheet_name)
    # label_data = label_data[label_data[INTENT_DESC].isin(cat_dim[INTENT_DESC])]
    label_data[LABEL] = 0

    # 保存数据
    label_data_source_file = get_file_path_without_suffix(label_data_file_list[0]) + NEG_LABEL_DATA_SOURCE_SUFFIX
    with pd.ExcelWriter(label_data_source_file) as writer:
        label_data.to_excel(writer, index=False)

    # 输出缺失数据类别
    lack_cat_dim_file = get_file_path_without_suffix(label_data_source_file) + NEG_CAT_NOT_IN_DATA_SUFFIX
    DataProcess.get_diff_cat(label_data, cat_dim, lack_cat_dim_file, min_count=1)


def get_test_sample_from_label_source(label_source_file_list, emotion_dim_map_file, intention_dim_map_file):
    '''
    采样生成测评样本数据：从标注集合中按照一定比比例采样各个类别的样本数据
    '''
    # 加载标注数据
    label_data_list = []
    for label_data_file in label_source_file_list:
        label_data_list.append(load_data2df(label_data_file))
    label_data = pd.concat(label_data_list)
    emotion_dim_map = load_data2df(emotion_dim_map_file)
    intention_dim_map = load_data2df(intention_dim_map_file)
    # 正样例中，各类采样10个sample
    pos_max_count = 10
    pos_label_data = label_data[label_data[LABEL] > 0].groupby([INTENT_DESC]).head(pos_max_count)

    neg_max_count = 5
    online_desc = pd.concat([emotion_dim_map[EMOTION_DESC], intention_dim_map[INTENT_DESC_96]])
    neg_label_data = label_data[label_data[INTENT_DESC].isin(online_desc) & label_data[LABEL] == 0]\
        .groupby([INTENT_DESC]).head(5)

    random_neg_label_count = pos_label_data.shape[0] - neg_label_data.shape[0]
    random_neg_label_data = label_data[~label_data[INTENT_DESC].isin(online_desc)]\
        .sample(n=random_neg_label_count)

    sample_data = pd.concat([pos_label_data, neg_label_data, random_neg_label_data])
    emotion_dim_map_dict = emotion_dim_map.set_index(EMOTION_DESC)[INTENT_DESC].to_dict()
    intention_dim_map_dict = intention_dim_map.set_index(INTENT_DESC_96)[INTENT_DESC].to_dict()
    sample_data[INTENT_DESC].replace(emotion_dim_map_dict, inplace=True)
    sample_data[INTENT_DESC].replace(intention_dim_map_dict, inplace=True)

    # 保存数据
    label_data_source_file = get_file_path_from_path(label_source_file_list[0]) + '\\' + TEST_FILE_PREFIX + f'_{pos_max_count}_{neg_max_count}.xlsx'
    with pd.ExcelWriter(label_data_source_file) as writer:
        sample_data.to_excel(writer, index=False)



if __name__ == '__main__':
    '''
        # Task0 加载训练数据源,输出缺失cat dim, 将cat转为线上已有的cat，用于拉取数据，进行标注
    '''
    # data_dir = "C:\\Users\\User\\Documents\\project\\intention\\data\\test\\"
    data_dir = "D:\\project\\intention\\test\\"
    label_data_list = [
        data_dir + "General_Intention_134_v1.xlsx"
    ]
    dim_dir = data_dir + 'dim\\'
    cat_dim_online = dim_dir + "intention_134_cat_dim_new.csv"
    cat_need_label = merge_pos_label_source_data(label_data_list, cat_dim_online, min_count=100)

    emotion_dim_file = dim_dir + 'intent43_intent134.xlsx'
    intention_dim_file = dim_dir + 'intent96_intent134.xlsx'
    intention_old_new_dim_map_file = dim_dir + 'intention_old_new_dim_map.xlsx'
    cat_need_label_online = convert_cat_desc(cat_need_label, intention_old_new_dim_map_file, is_map_reverse=True)
    print(cat_need_label.to_string())
    cat_need_label_online_out_file = get_file_path_without_suffix(label_data_list[0]) + CAT_NOT_IN_DATA_SUFFIX
    with pd.ExcelWriter(cat_need_label_online_out_file) as writer:
        cat_need_label_online.to_excel(writer, index=False)
    # merge_neg_label_source_data(label_data_list, cat_dim_online)

    '''
    # Task1 加载已有数据源,输出 label_data_source 及 缺失cat dim
    '''
    # data_dir = "C:\\Users\\User\\Documents\\project\\intention\\data\\test\\"
    data_dir = "D:\\project\\intention\\test\\"
    label_data_list = [
        data_dir + "intention_test_labeled_data_all.xlsx",
        data_dir + "正向情绪index标签分类.xlsx",
        data_dir + "intention_emotion_last7day.labeled.xlsx",
        data_dir + "intention_64_1025-1031.sample.xlsx",
        data_dir + "intention_19_1018-1024.sample.xlsx",
        data_dir + "intention_4_1004-1010.sample.xlsx",
        data_dir + "intention_label_manual.xlsx",
        data_dir + "intention_neg_1106-1113.sample.xlsx"

    ]
    cat_dim_online = data_dir + "intention_134_cat_dim_online.csv"
    # merge_pos_label_source_data(label_data_list, cat_dim_online)
    # merge_neg_label_source_data(label_data_list, cat_dim_online)

    '''
    # TASK2: 从无标日志中采样待标注样本, output: 待标注数据
    '''
    log_data_dir = data_dir
    # log_file_path = log_data_dir + "intention_emotion_last7day_1.csv"
    log_file_path = log_data_dir + "intention_neg_1106-1113.csv"
    max_per_cat = 15
    cat_dim_file = data_dir + "intention_test_labeled_data_all.neg_label_data_source.neglackcat.xlsx"
    # file_for_annotation = data_dir + 'not_in_dim_sample.xlsx'
    # sample_data_from_log(log_file_path, max_per_cat, cat_dim_file=cat_dim_file)


    # 采样测评数据
    label_source_dir = data_dir + 'sample\\'
    label_source_file_list = [
        label_source_dir + 'intention_test_labeled_data_all.label_data_source.xlsx',
        label_source_dir + 'intention_test_labeled_data_all.neg_label_data_source.xlsx'
    ]
    dim_dir = data_dir + 'dim\\'
    emotion_dim_file = dim_dir + 'intent43_intent134.xlsx'
    intention_dim_file = dim_dir + 'intent96_intent134.xlsx'
    # get_test_sample_from_label_source(label_source_file_list, emotion_dim_file, intention_dim_file)



    # label_file = data_dir + "IntentionDetection_汇总.xlsx"
    # cat_info_file = data_dir + "queryintention.response.leilei.txt"
    # # DataProcess.trans_label2sample(label_file, label_file+".sample.xlsx", 1500)
    # DataProcess.cat_diff(cat_info_file, label_file)


    '''
    # TASK1:  统计分析测评文件中类别分布、五样本类别、误样本类别、完成emotion和intention类别映射
    test_data_dir = data_dir
    test_label_data = test_data_dir + "intention_test_labeled_data.xlsx"
    emotion_intention_map_file = test_data_dir + "emotion_intention_map.txt"
    intention_cat_dim_file = test_data_dir + "intention_134_cat_dim_online.txt"
    test_data_describe(test_label_data, emotion_intention_map_file, intention_cat_dim_file)

    # TASK2: 从无标日志中采样标注样本
    log_data_dir = data_dir
    log_file_path = log_data_dir + "intention_emotion_last7day_1.csv"
    # log_file_path = log_data_dir + "intention_emotion_last7day_all.csv"
    max_per_cat = 15
    cat_dim_file = data_dir + "intention_test_labeled_data.xlsx.analysis.xlsx"
    out_file_path = 'not_in_dim_sample.xlsx'
    sample_data_from_log(log_file_path, max_per_cat, cat_dim_file=cat_dim_file, sheet_name_of_dim_file=LACK_CAT_SHEET, out_file_path=out_file_path)

    log_data_sample_file = get_file_path_without_suffix(log_file_path) + SAMPLE_SUFFIX
    cat_not_data_file = get_file_path_without_suffix(log_data_sample_file) + CAT_NOT_IN_DATA_SUFFIX
    DataProcess.get_diff_cat_from_file(log_data_sample_file, cat_dim_file, sheet_name_of_dim_file=LACK_CAT_SHEET, out_file=cat_not_data_file)
    '''






