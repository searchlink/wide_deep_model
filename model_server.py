# -*- coding: utf-8 -*-
# @Time    : 2019/4/27 15:45
# @Author  : skydm
# @Email   : wzwei1636@163.com
# @File    : model_server.py
# @Software: PyCharm

import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 保证CUDA使用的设备ID和硬件的ID保持一致
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import pandas as pd
import tensorflow as tf
import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow_serving.apis import classification_pb2
from tensorflow.python.framework import tensor_util


# tf.contrib.predictor.from_saved_model 方法能够将导出的模型加载进来，直接生成一个预测函数供使用
def prepare_examples_for_prediction():
    '''为预测构造输入, 指定column的封装类型，且输入封装成Example协议'''
    predict_path = "/tmp/pycharm_project_717/rec_sys/recall_strategy/wide_deep-hh/data/predict_fbt_data.csv"
    df = pd.read_csv(predict_path)
    # df = pd.read_csv("../data/predict_fbt_data.csv")

    reserved_columns = ['sess_id', 'r_pid', 'dctime', 'customer_id', 'categories_id', 'product_id', 'user_id', 'click',
                        'level_2', 'products_price', 'products_cost_price', 'itemcf_sim',
                        'title_sim', 'tag_sim', 'fbt_show_count', 'fbt_view2_count', 'fbt_click_rate', 'fbt_view_count',
                        'fbt_wish_count', 'fbt_cart_count', 'fbt_order_count', 'fbt_addcart_rate', 'fbt_order_rate',
                         'country_code', 'operating_system', 'browser', 'browser_version',
                        'utm_source', 'gender', 'age_tag', 'has_child', 'payment_method', 'price_tendency',
                        ]

    keep_columns = ['click', 'categories_id', 'level_2', 'products_price', 'products_cost_price', 'itemcf_sim',
                    'title_sim', 'tag_sim', 'fbt_show_count', 'fbt_view2_count', 'fbt_click_rate', 'fbt_view_count',
                    'fbt_wish_count', 'fbt_cart_count', 'fbt_order_count', 'fbt_addcart_rate', 'fbt_order_rate',
                    'country_code', 'operating_system', 'browser', 'browser_version', 'utm_source', 'gender', 'age_tag',
                    'has_child', 'payment_method', 'price_tendency']
    num_records = len(df.index)
    keeped_df = df.loc[:, keep_columns]
    rows_to_predict = keeped_df.iloc[0:num_records]
    # 将pandas的每一行转化成一个字典  examples_dict is list
    examples_dict = rows_to_predict.to_dict("records")  # class: {'dict', 'list', 'series', 'split', 'records', 'index'}

    examples = []

    print("开始数据处理：")
    # 构造每个样本的Example协议块
    for example_dict in examples_dict:
        # 按照数据类型进行feature封装
        feature_dict = {}
        for key in keep_columns:
            # 注意gender和has_child需要在pandas中进行转化astype(int)
            if key in ["categories_id", "level_2", "gender", "has_child"]:
                # 进行格式转化, 类别型变量且整数类型转化为int64
                feature_dict[key] = tf.train.Feature(int64_list=tf.train.Int64List(value=[example_dict[key]]))
            elif key in ["browser", 'age_tag', 'price_tendency', "operating_system",
                         "country_code", "browser_version", "utm_source", "payment_method"]:
                # 进行格式转化, 类别型变量且字符串类型转化为byte
                feature_dict[key] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[example_dict[key].encode()]))
            else:
                feature_dict[key] = tf.train.Feature(float_list=tf.train.FloatList(value=[example_dict[key]]))

        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        examples.append(example)

    return examples

def query(channel, examples):

    # 发送请求
    # request = predict_pb2.PredictRequest()
    request = classification_pb2.ClassificationRequest()
    '''
    tensorflow_model_server --port=9000 --model_name=wide_deep --model_base_path=/tmp/pycharm_project_717/
    rec_sys/recall_strategy/wide_deep-hh/wide_deep_model/model/wide_deep_64_40_2_1556356457

    signature_def['serving_default']:
      The given SavedModel SignatureDef contains the following input(s):
        inputs['inputs'] tensor_info:
            dtype: DT_STRING
            shape: (-1)
            name: input_example_tensor:0
      The given SavedModel SignatureDef contains the following output(s):
        outputs['classes'] tensor_info:
            dtype: DT_STRING
            shape: (-1, 2)
            name: head/Tile:0
        outputs['scores'] tensor_info:
            dtype: DT_FLOAT
            shape: (-1, 2)
            name: head/predictions/probabilities:0
      Method name is: tensorflow/serving/classify

    '''
    request.model_spec.name = "wide_deep"  # the model name used above when starting TF serving
    request.model_spec.signature_name = "serving_default"  # get this from Saved Model CLI output
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    request.input.example_list.examples.extend(examples)
    response = stub.Classify(request, 10.0) # 10 is the timeout in seconds, but its blazing fast
    print(dir(response))
    predictions = list(response.result.classifications)
    '''
    单个example输出记录：
    classes {
        label: "0"
        score: 0.46758782863616943
    }
    classes {
        label: "1"
        score: 0.5324121713638306
    }
    # predictions[0].classes
    [label: "0"
        score: 0.7774916887283325
    , label: "1"
        score: 0.22250832617282867
    ]
    '''
    prob = [round(predictions[i].classes[1].score, 4) for i in range(len(predictions))]
    return prob

if __name__ == '__main__':
    # 声明全局
    channel = grpc.insecure_channel('192.168.15.27:9000')

    examples = prepare_examples_for_prediction()
    num_samples = len(examples)

    batch_size = 100
    num_batches = num_samples // batch_size

    # 预测为1的概率结果列表
    predict_one_prob = []
    for i in range(num_batches):
        print('batch {}'.format(i))
        batch_examples = examples[i*batch_size: (i+1)*batch_size]
        prob = query(channel, batch_examples)
        predict_one_prob += prob

    print("完成预测！")
