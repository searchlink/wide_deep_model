# -*- coding: utf-8 -*-
# @Time    : 2019/4/26 10:50
# @Author  : skydm
# @Email   : wzwei1636@163.com
# @File    : wide_deep_fbt.py
# @Software: PyCharm

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 保证CUDA使用的设备ID和硬件的ID保持一致
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
import sys
import time
import shutil

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default="wide_deep",
                    help="选择要使用的模型：{'wide', 'deep', 'wide_deep'}")
parser.add_argument('--train_epoch', type=int, default=40,
                    help="训练的迭代次数")
parser.add_argument('--epoch_per_eval', type=int, default=2,
                    help="每次评估需要迭代的次数")
parser.add_argument('--batch_size', type=int, default=64,
                    help="每次批量迭代的批次数")
parser.add_argument('--train_data', type=str, default="/tmp/pycharm_project_717/rec_sys/recall_strategy/wide_deep-hh/data/train_fbt_data.csv",
                    help="训练数据的文件路径")
parser.add_argument('--test_data', type=str, default="/tmp/pycharm_project_717/rec_sys/recall_strategy/wide_deep-hh/data/test_fbt_data.csv",
                    help="测试数据的文件路径")

_CSV_COLUMNS_NAME = ['click', 'categories_id', 'level_2', 'products_price', 'products_cost_price', 'itemcf_sim',
                        'title_sim', 'tag_sim', 'fbt_show_count', 'fbt_view2_count', 'fbt_click_rate', 'fbt_view_count',
                        'fbt_wish_count', 'fbt_cart_count', 'fbt_order_count', 'fbt_addcart_rate', 'fbt_order_rate',
                        'country_code', 'operating_system', 'browser', 'browser_version', 'utm_source', 'gender',
                        'age_tag', 'has_child', 'payment_method', 'price_tendency']


# 因为输入的数据已经进行了预处理
_CSV_COLUMN_DEFAULTS = [[0], [0], [0], [0.0], [0.0], [0.0],
                        [0.0], [0.0], [0], [0], [0.0], [0],
                        [0], [0], [0], [0.0], [0.0],
                        [''], [''], [''], [''], [''], [0],
                        [''], [0], [''], ['']]

_NUM_EXAMPLES = {
    'train': 57802,
    'validation': 14450,
}

import tensorflow as tf

# 定义tensorflow的运行配置信息
def get_session():
    cfg = tf.ConfigProto(log_device_placement=False) # 获取到 operations 和 Tensor 被指派到哪个设备
    cfg.gpu_options.allow_growth = True # 程序用多少就占多少内存
    return tf.Session(config=cfg)

sess = get_session()

def build_feature_column():
    """
    建立wide和deep feature columns
    """
    # continue column
    products_price = tf.feature_column.numeric_column("products_price")
    products_cost_price = tf.feature_column.numeric_column("products_cost_price")
    itemcf_sim = tf.feature_column.numeric_column("itemcf_sim")
    title_sim = tf.feature_column.numeric_column("title_sim")
    tag_sim = tf.feature_column.numeric_column("tag_sim")
    fbt_show_count = tf.feature_column.numeric_column("fbt_show_count")
    fbt_view2_count = tf.feature_column.numeric_column("fbt_view2_count")
    fbt_click_rate = tf.feature_column.numeric_column("fbt_click_rate")
    fbt_view_count = tf.feature_column.numeric_column("fbt_view_count")
    fbt_wish_count = tf.feature_column.numeric_column("fbt_wish_count")
    fbt_cart_count = tf.feature_column.numeric_column("fbt_cart_count")
    fbt_order_count = tf.feature_column.numeric_column("fbt_order_count")
    fbt_addcart_rate = tf.feature_column.numeric_column("fbt_addcart_rate")
    fbt_order_rate = tf.feature_column.numeric_column("fbt_order_rate")

    # 连续特征离散化
    price_buckets = tf.feature_column.bucketized_column(products_price, boundaries=[5, 10, 30, 100])

    # categorical_column_with_vocabulary_list
    browser = tf.feature_column.categorical_column_with_vocabulary_list("browser", vocabulary_list=['chrome', 'firefox', 'safari', 'other', 'ie'])
    gender = tf.feature_column.categorical_column_with_vocabulary_list("gender", vocabulary_list=[0, -1, 1])
    age_tag = tf.feature_column.categorical_column_with_vocabulary_list('age_tag', vocabulary_list=['36~56', '0~35', 'other', '56~100'])
    has_child = tf.feature_column.categorical_column_with_vocabulary_list('has_child', vocabulary_list=[1, -1, 0])
    price_tendency = tf.feature_column.categorical_column_with_vocabulary_list('price_tendency', vocabulary_list=['other', 'mid', 'high', 'low'])
    operating_system = tf.feature_column.categorical_column_with_vocabulary_list("operating_system", vocabulary_list=['win', 'mac', 'linux', 'other', 'ipad', 'sunos', 'iphone'])

    # categorical_column_with_hash_bucket(我愿意为我的输入设置多少类别)
    # when assign data type tf.int32, Value passed to parameter 'sparse_types' has DataType int32 not in list of allowed values: str, int64
    categories_id = tf.feature_column.categorical_column_with_hash_bucket("categories_id", hash_bucket_size=1000, dtype=tf.int64)
    level_2 = tf.feature_column.categorical_column_with_hash_bucket("level_2", hash_bucket_size=50, dtype=tf.int64)
    country_code = tf.feature_column.categorical_column_with_hash_bucket("country_code", hash_bucket_size=30)
    browser_version = tf.feature_column.categorical_column_with_hash_bucket("browser_version", hash_bucket_size=50)
    utm_source = tf.feature_column.categorical_column_with_hash_bucket("utm_source", hash_bucket_size=50)
    payment_method = tf.feature_column.categorical_column_with_hash_bucket("payment_method", hash_bucket_size=25)

    # wide模型的特征都是离散特征、离散特征之间的交互作用特征
    # deep模型的特征则是离散特征embedding 加上连续特征；
    # wide端模型通过离散特征的交叉组合进行memorization; deep 端模型通过特征的embedding进行generalization
    # 注意： tf.feature_column.crossed_column的key必须是字符串, 且不能使用hashed categorical column

    class_columns = [price_buckets, browser, gender, age_tag, has_child, price_tendency, operating_system, categories_id, level_2, country_code, browser_version, utm_source, payment_method]
    crossed_columns = [
        tf.feature_column.crossed_column(["gender", "categories_id"], hash_bucket_size=1000),
        tf.feature_column.crossed_column(["age_tag", "operating_system"], hash_bucket_size=1000),
        tf.feature_column.crossed_column(["age_tag", "payment_method"], hash_bucket_size=1000),
        tf.feature_column.crossed_column(["age_tag", "operating_system", "payment_method"], hash_bucket_size=1000),
    ]

    wide_columns = class_columns


    # indicator_column的作用就是将category产生的sparser tensor转换成dense tensor.
    # Sparse Features -> Embedding vector -> 串联(Embedding vector, 连续特征) -> 输入到Hidden Layer
    # embedding_column(embedding);indicator_column(multi-hot);
    deep_columns = [products_price, products_cost_price, itemcf_sim, title_sim, tag_sim,
                    fbt_show_count, fbt_view2_count, fbt_click_rate, fbt_view_count, fbt_wish_count,
                    fbt_cart_count, fbt_order_count, fbt_addcart_rate, fbt_order_rate,
                    tf.feature_column.indicator_column(gender),
                    tf.feature_column.indicator_column(age_tag),
                    tf.feature_column.indicator_column(has_child),
                    tf.feature_column.indicator_column(browser),
                    tf.feature_column.indicator_column(price_tendency),
                    tf.feature_column.indicator_column(operating_system),
                    tf.feature_column.indicator_column(categories_id),
                    tf.feature_column.indicator_column(level_2),
                    tf.feature_column.indicator_column(country_code),
                    tf.feature_column.indicator_column(browser_version),
                    tf.feature_column.indicator_column(utm_source),
                    tf.feature_column.indicator_column(payment_method),
                    tf.feature_column.embedding_column(categories_id, dimension=8)
                    ]

    return wide_columns, deep_columns

# 读取数据
def tf_read_data(file):
    assert tf.gfile.Exists(file), print("{} is not found".format(file))
    dataset = tf.data.TextLineDataset(file) # 每一个元素对应一行
    return dataset

# 构造训练模型的输入
def input_fn(dataset, num_epochs, shuffle, batch_size):
    """Generate an input function for the Estimator."""
    def parse_csv(value):

        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS_NAME, columns))
        labels = features.pop('click')
        return features, labels

    if shuffle:
        dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

    dataset = dataset.map(parse_csv, num_parallel_calls=5)

    # We call repeat after shuffling, rather than before, to prevent separate epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels

# 定义使用的模型
def build_estimator(model_dir, model_type, run_config):
    """为指定的模型类型构建estimator"""
    # 生成feature col
    wide_columns, deep_columns = build_feature_column()
    hidden_units = [100, 50]

    if model_type == "wide":
        model = tf.estimator.LinearClassifier(model_dir=model_dir, feature_columns=wide_columns, config=run_config)
    elif model_type == "deep":
        model = tf.estimator.DNNClassifier(model_dir=model_dir, feature_columns=deep_columns, hidden_units=hidden_units, config=run_config)
    else:
        model = tf.estimator.DNNLinearCombinedClassifier(model_dir=model_dir, linear_feature_columns=wide_columns,
                                                         dnn_feature_columns=deep_columns, dnn_hidden_units=hidden_units, config=run_config)
    return model

def main(unused_argv):
    print("FLAG: ", FLAG)
    print("unused_argv: ", unused_argv)

    cur_model_dir = "model/{}_{}_{}_{}_{}".format(FLAG.model_type, FLAG.batch_size, FLAG.train_epoch,
                                                  FLAG.epoch_per_eval, str(int(time.time())))
    shutil.rmtree(cur_model_dir, ignore_errors=True)
    cfg = tf.ConfigProto(log_device_placement=False)
    cfg.gpu_options.allow_growth = True

    # Estimator类似sklearn模块，提供高阶应用
    # keep_checkpoint_max 保留最新的checkpoint
    # save_summary_steps每隔这么多步骤保存摘要,用于绘制图像
    # save_checkpoints_steps 每隔多少步保存检查点
    # log_step_count_steps 每训练多少次输出一次损失值
    run_cfg = tf.estimator.RunConfig().replace(session_config=cfg,
                                               keep_checkpoint_max=1,
                                               save_summary_steps=10000,
                                               save_checkpoints_steps=10000,
                                               log_step_count_steps=10000)

    model = build_estimator(cur_model_dir, FLAG.model_type, run_cfg)

    # 读取训练数据
    train_data = tf_read_data(FLAG.train_data)
    test_data = tf_read_data(FLAG.test_data)
    print("训练数据和测试数据已经读入！")

    # 开始进行训练
    for i in range(FLAG.train_epoch // FLAG.epoch_per_eval):
        start_time = time.clock()  # 统计CPU的运行时间
        print("-" * 60)
        print("# eval: ", str(i+1))
        # model.train的参数input_fn是函数
        model.train(input_fn=lambda: input_fn(train_data, FLAG.epoch_per_eval, True, FLAG.batch_size))
        end_time = time.clock()
        print("平均每个epoch花费时间：{}".format(int((end_time - start_time)/FLAG.epoch_per_eval)))

        print("*" * 20, "开始进行评估：", "*" * 20)
        results = model.evaluate(input_fn=lambda: input_fn(test_data, 1, False, FLAG.batch_size))
        print("# epoch_{} result: ".format((i+1)*FLAG.epoch_per_eval))

        for key in sorted(results):
            print("%s : %s" % (key, results[key]))

        print("-" * 60)

    # 开始保存模型，为后续提供server服务(需要定义导出目录，用于模型的接收参数)
    wide_columns, deep_columns = build_feature_column()
    print("模型的输入列名：", wide_columns, deep_columns)
    features_spec = tf.feature_column.make_parse_example_spec(wide_columns + deep_columns)
    print("从输入列名开始创建字典！ ")

    # 构建接收函数，并导出模型
    export_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(features_spec)
    print("已构建接收参数")
    model.export_savedmodel(cur_model_dir, export_input_fn)
    print("模型已导出！")


if __name__ == '__main__':
    # 设置日志的可视化级别
    tf.logging.set_verbosity(tf.logging.ERROR)
    FLAG, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]]+unparsed)