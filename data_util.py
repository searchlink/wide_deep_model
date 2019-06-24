# -*- coding: utf-8 -*-
# @Time    : 2019/4/25 18:02
# @Author  : skydm
# @Email   : wzwei1636@163.com
# @File    : data_util.py
# @Software: PyCharm

import os
import pandas as pd

# 用户在点击位的行为
data_path = os.path.join(os.getcwd(), 'rec_sys/recall_strategy/wide_deep-hh/data/fbt_features.csv')
df = pd.read_csv(data_path)

print(df.shape)
print(df.head())
print(df.dtypes)

columns_name = df.columns.tolist()

# 保留的列名
reversed_columns = ['click', 'categories_id', 'level_2', 'products_price', 'products_cost_price', 'itemcf_sim',
                    'title_sim', 'tag_sim', 'fbt_show_count', 'fbt_view2_count', 'fbt_click_rate', 'fbt_view_count',
                    'fbt_wish_count', 'fbt_cart_count', 'fbt_order_count', 'fbt_addcart_rate', 'fbt_order_rate',
                    # 'fbt_wish_rate', 'all_site_order_count', 'all_site_cart_count', 'all_site_view_count',
                    # 'all_site_show_count', 'all_site_order_rate', 'all_site_addcart_rate', 'all_site_click_rate',
                    'country_code', 'operating_system', 'browser', 'browser_version', 'utm_source', 'gender', 'age_tag',
                    'has_child', 'payment_method', 'price_tendency']

# 保留的数据
reversed_df = df[reversed_columns]

cleaned_df = reversed_df.drop_duplicates()

# 填充数据
cleaned_df[['operating_system', 'browser', 'browser_version', 'utm_source', 'gender', 'age_tag',
                    'has_child', 'payment_method', 'price_tendency']].head()

cleaned_df['country_code'].fillna('other', inplace=True)
cleaned_df["operating_system"].fillna("other", inplace=True)
cleaned_df["browser"].fillna('other', inplace=True)
cleaned_df["browser_version"].fillna('other', inplace=True)
cleaned_df["utm_source"].fillna('other', inplace=True)
cleaned_df["age_tag"].fillna('other', inplace=True)
cleaned_df["payment_method"].fillna('other', inplace=True)
cleaned_df["price_tendency"].fillna('other', inplace=True)

cleaned_df["has_child"].fillna(-1, inplace=True)
cleaned_df["gender"].fillna(-1, inplace=True)

# 类型推断,例如 float64 to int64 if possible
cleaned_df.fillna(0, inplace=True, downcast="infer")

print(cleaned_df.dtypes)

# 划分训练集和测试集
train_df = cleaned_df.sample(frac=0.8, random_state=123)
test_df = cleaned_df.drop(index=train_df.index)

# 保存数据到本地
train_path = os.path.join(os.getcwd(), 'rec_sys/recall_strategy/wide_deep-hh/data/train_fbt_data.csv')
test_path = os.path.join(os.getcwd(), 'rec_sys/recall_strategy/wide_deep-hh/data/test_fbt_data.csv')
train_df.to_csv(train_path, header=None, index=False)
test_df.to_csv(test_path, header=None, index=False)

#-----------------------------------------------------------------------------------------------------------
data_path = os.path.join(os.getcwd(), 'rec_sys/recall_strategy/wide_deep-hh/data/fbt_features.csv')
cleaned_df = pd.read_csv(data_path)
cleaned_df = cleaned_df.drop_duplicates()

# 填充数据
cleaned_df[['operating_system', 'browser', 'browser_version', 'utm_source', 'gender', 'age_tag',
                    'has_child', 'payment_method', 'price_tendency']].head()

cleaned_df['country_code'].fillna('other', inplace=True)
cleaned_df["operating_system"].fillna("other", inplace=True)
cleaned_df["browser"].fillna('other', inplace=True)
cleaned_df["browser_version"].fillna('other', inplace=True)
cleaned_df["utm_source"].fillna('other', inplace=True)
cleaned_df["age_tag"].fillna('other', inplace=True)
cleaned_df["payment_method"].fillna('other', inplace=True)
cleaned_df["price_tendency"].fillna('other', inplace=True)

cleaned_df["has_child"].fillna(-1, inplace=True)
cleaned_df["gender"].fillna(-1, inplace=True)
cleaned_df["has_child"] = cleaned_df["has_child"].astype(int)
cleaned_df["gender"] = cleaned_df["gender"].astype(int)

float_columns = ['click', 'products_price', 'products_cost_price', 'itemcf_sim',
                    'title_sim', 'tag_sim', 'fbt_show_count', 'fbt_view2_count', 'fbt_click_rate', 'fbt_view_count',
                    'fbt_wish_count', 'fbt_cart_count', 'fbt_order_count', 'fbt_addcart_rate', 'fbt_order_rate']

# 类型推断,例如 float64 to int64 if possible
cleaned_df['click'].fillna(0, inplace=True, downcast="infer")
cleaned_df['products_price'].fillna(0, inplace=True, downcast="infer")
cleaned_df['products_cost_price'].fillna(0, inplace=True, downcast="infer")
cleaned_df['itemcf_sim'].fillna(0, inplace=True, downcast="infer")
cleaned_df['title_sim'].fillna(0, inplace=True, downcast="infer")
cleaned_df['tag_sim'].fillna(0, inplace=True, downcast="infer")
cleaned_df['fbt_show_count'].fillna(0, inplace=True, downcast="infer")
cleaned_df['fbt_view2_count'].fillna(0, inplace=True, downcast="infer")
cleaned_df['fbt_click_rate'].fillna(0, inplace=True, downcast="infer")
cleaned_df['fbt_view_count'].fillna(0, inplace=True, downcast="infer")
cleaned_df['fbt_wish_count'].fillna(0, inplace=True, downcast="infer")
cleaned_df['fbt_cart_count'].fillna(0, inplace=True, downcast="infer")
cleaned_df['fbt_order_count'].fillna(0, inplace=True, downcast="infer")
cleaned_df['fbt_addcart_rate'].fillna(0, inplace=True, downcast="infer")
cleaned_df['fbt_order_rate'].fillna(0, inplace=True, downcast="infer")

predict_path = os.path.join(os.getcwd(), 'rec_sys/recall_strategy/wide_deep-hh/data/predict_fbt_data.csv')
predict_sample = cleaned_df.sample(1000, random_state=1)
predict_sample.to_csv(predict_path, index=False)




