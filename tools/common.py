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


def convert_cat_desc(cat_desc_list, cat_map_dim_file, is_map_reverse=False, ):
    '''
    将cat_list 转为 cat_map
    '''
    cat_dim = load_data2df(cat_map_dim_file)
    if is_map_reverse:
        # 转换map映射关系
        cols = list(cat_dim)
        cols.insert(0, cols.pop(1))
    cat_dim_dict = cat_dim.set_index(cols[0])[cols[1]].to_dict()
    cat_desc_convert = cat_desc_list.replace(cat_dim_dict)
    return cat_desc_convert


def load_data2df(input_file, sheet_name=0, seq='\t'):
    '''
    使用pandas加载数据
    '''
    try:
        suffix = get_file_suffix_from_path(input_file)
        if suffix in ["xls", 'xlsx']:
            input_pd = pd.read_excel(input_file, sheet_name=sheet_name)
        elif "csv" == suffix:
            input_pd = pd.read_csv(input_file)
    except Exception:
        input_pd = pd.read_table(input_file, seq)

    return input_pd


def get_file_name_no_suffix_from_path(file_path):
    file_name = file_path[file_path.rfind("\\") + 1: file_path.rfind(".")]
    return file_name


def get_file_name_with_suffix_from_path(file_path):
    file_name = file_path[file_path.rfind("\\") + 1:]
    return file_name


def get_file_path_from_path(file_path):
    if file_path.rfind("\\") == len(file_path) - 1:
        return file_path
    data_dir = file_path[0: file_path.rfind("\\")]
    return data_dir


def get_file_suffix_from_path(file_path):
    file_suffix = file_path[file_path.rfind(".") + 1:]
    return file_suffix


def get_file_path_without_suffix(file_path):
    file_path = file_path[:file_path.rfind(".")]
    return file_path
