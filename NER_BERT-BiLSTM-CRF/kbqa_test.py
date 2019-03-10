#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import sys
import re
import time
import jieba
import numpy as np
import pandas as pd
import urllib.request
import urllib.parse
import tensorflow as tf
from db import load_data_kudu
from global_config import Logger

sys.path.append('/home/mqq/zwshi/bert/')
from similarity import BertSim
# 模块导入 https://blog.csdn.net/xiongchengluo1129/article/details/80453599

loginfo = Logger("recommend_articles.log", "info")
file = "./NERdata/q_t_a_testing_predict.txt"

bs = BertSim()
bs.set_mode(tf.estimator.ModeKeys.PREDICT)


def dataset_test():
    '''
    用训练问答对中的实体+属性，去知识库中进行问答测试准确率上限
    :return:
    '''
    with open(file) as f:
        total = 0
        recall = 0
        correct = 0

        for line in f:
            question, entity, attribute, answer, ner = line.split("\t")
            ner = ner.replace("#", "").replace("[UNK]", "%")
            # case1: entity and attribute Exact Match
            sql_e1_a1 = "select * from ods_logs.ods_profile_kb_temp where entity='"+entity+"' and attribute='"+attribute+"' limit 10"
            result_e1_a1 = load_data_kudu.load_data(sql_e1_a1, True)
            # case2: entity Fuzzy Match and attribute Exact Match
            sql_e0_a1 = "select * from ods_logs.ods_profile_kb_temp where entity like '%" + entity + "%' and attribute='" + attribute + "' limit 10"
            #result_e0_a1 = load_data_kudu.load_data(sql_e0_a1, True)
            # case3: entity Exact Match and attribute Fuzzy Match
            sql_e1_a0 = "select * from ods_logs.ods_profile_kb_temp where entity like '" + entity + "' and attribute='%" + attribute + "%' limit 10"
            #result_e1_a0 = load_data_kudu.load_data(sql_e1_a0, True)

            if len(result_e1_a1) > 0:
                recall += 1
                for l in result_e1_a1:
                    if l[2] == answer:
                        correct += 1
            else:
                result_e0_a1 = load_data_kudu.load_data(sql_e0_a1, True)
                if len(result_e0_a1) > 0:
                    recall += 1
                    for l in result_e0_a1:
                        if l[2] == answer:
                            correct += 1
                else:
                    result_e1_a0 = load_data_kudu.load_data(sql_e1_a0, True)
                    if len(result_e1_a0) > 0:
                        recall += 1
                        for l in result_e1_a0:
                            if l[2] == answer:
                                correct += 1
                    else:
                        loginfo.logger.info(sql_e1_a0)
            if total > 100:
                break
            total += 1
            time.sleep(1)
            loginfo.logger.info("total: {}, recall: {}, correct:{}, accuracy: {}%".format(total, recall, correct, correct * 100.0 / recall))
        #loginfo.logger.info("total: {}, recall: {}, correct:{}, accuracy: {}%".format(total, recall, correct, correct*100.0/recall))


def load_embedding(path):
    embedding_index = {}
    f = open(path, encoding='utf8')
    for index, line in enumerate(f):
        if index == 0:
            continue
        values = line.split(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs
    f.close()

    return embedding_index


jieba.load_userdict('./NLPCC2016KBQA/kbqa_attribute_dict.txt')
# jieba.analyse.set_stop_words('./nlp/stopwords.txt')
# jieba.analyse.set_stop_words('./nlp/stopwords_csdn.txt')
# stoplist = [w.strip() for w in io.open('./NLPCC2016KBQA/stopwords.txt', encoding="utf-8")]
stoplist = []
stoplist = stoplist+[w.strip() for w in io.open('./NLPCC2016KBQA/stopwords_csdn.txt', encoding="utf-8")]
loginfo.logger.info("stoplist size:%s" % len(stoplist))


def get_word_list_str(doc):
    '''
    jieba 分词结果，格式： 我\t喜欢\t你
    :param doc:
    :return:
    '''
    #return '|'.join(jieba.cut(doc))
    return jieba.cut(doc)


def cosine_similarity(vector1, vector2):
    try:
        return np.dot(vector1, vector2)/(np.linalg.norm(vector1) * np.linalg.norm(vector2))
    except Exception as e:
        loginfo.logger.info("error cosine_similarity %s" % repr(e))


def get_wrod_vector(word_str):
    '''

    :param word_str:  AAA,BBB,CCC
    :return:
    '''
    w2v = np.zeros(200)
    try:
        params = urllib.parse.urlencode({'word': word_str})
        # loginfo.logger.info("url: http://192.168.9.91:8011/tencent_model?%s" % params)
        f = urllib.request.urlopen("http://192.168.9.91:8011/tencent_model?%s" % params, timeout=60).read()
        result = eval(f.decode("utf-8"))
        if result['iRet'] != 0:
            return w2v
        else:
            for (k, v) in result.items():
                if k not in ("iRet", "desc"):
                    w2v += np.array(v)
            for k in word_str.split(","):
                if result.get(k) is None:
                    loginfo.logger.info("未知的单词：sentence: %s , word: %s" % (word_str, k))
    except Exception as e:
        loginfo.logger.info("error get_wrod_vector %s" % repr(e))
    return w2v


tencent_word_embedding = {}


def kb_test():
    '''
    进行问答测试：
    1、 实体检索
    2、 属性映射
    3、 答案组合
    :return:
    '''
    with open(file) as f:
        total = 0
        recall = 0
        correct = 0
        ambiguity = 0

        for line in f:
            question, entity, attribute, answer, ner = line.split("\t")
            ner = ner.replace("#", "").replace("[UNK]", "%").replace("\n", "")
            # case1: entity Exact Match
            sql_e1_a1 = "select * from ods_logs.ods_profile_kb_temp where entity='"+ner+"' limit 100"
            result_e1_a1 = load_data_kudu.load_data(sql_e1_a1, True)
            # case2: entity Fuzzy Match
            sql_e0_a1 = "select * from ods_logs.ods_profile_kb_temp where entity like '%" + ner + "%' order by length(entity) asc limit 20"
            #result_e0_a1 = load_data_kudu.load_data(sql_e0_a1, True)

            if len(result_e1_a1) > 0:
                recall += 1
                for l in result_e1_a1:
                    if l[1] in question:
                        if l[2] == answer:
                            correct += 1
                        else:
                            loginfo.logger.info("\t".join(l))
                            ambiguity += 1
                        break
            # 开始语义匹配
            else:
                result = get_word_list_str(question)
                result_split_stop = [w for w in result if not re.match('^[a-z|A-Z|0-9|.]*$', w) and w not in stoplist and w not in ner]
                #result_split_stop = ['名字', '是', '什么']
                loginfo.logger.info(result_split_stop)
                result_e0_a1 = load_data_kudu.load_data(sql_e0_a1, True)
                result_df = pd.DataFrame(result_e0_a1, columns=['entity', 'attribute', 'value'])
                loginfo.logger.info(result_df.head(100))

                if len(result_split_stop) < 1:
                    loginfo.logger.info("cannot fuzzy match entity: " + sql_e0_a1)
                    continue
                else:
                    question_word_vec = get_wrod_vector(",".join(result_split_stop))
                    #loginfo.logger.info(question_word_vec)
                    attribute_list = list(set(result_df["attribute"].tolist()))

                    params = urllib.parse.urlencode({'word': ",".join(attribute_list)})
                    # loginfo.logger.info("url: http://192.168.9.91:8011/tencent_model?%s" % params)
                    f = urllib.request.urlopen("http://192.168.9.91:8011/tencent_model?%s" % params).read()
                    attribute_vec = eval(f.decode("utf-8"))
                    if attribute_vec['iRet'] != 0:
                        loginfo.logger.info("cannot get the attribute vector")
                        continue

                    question_candicate_sim = [(k, cosine_similarity(v, question_word_vec)) for (k, v) in attribute_vec.items() if k not in ("iRet", "desc")]
                    question_candicate_sort = sorted(question_candicate_sim, key=lambda candicate: candicate[1], reverse=True)
                    loginfo.logger.info("\n".join([str(k)+" "+str(v) for (k, v) in question_candicate_sort]))

                    answer_candicate_df = result_df[result_df["attribute"] == question_candicate_sort[0][0]]
                    answer_candicate = answer_candicate_df["value"].tolist()
                    for answer_temp in answer_candicate:
                        if answer_temp == answer:
                            correct += 1
                        else:
                            loginfo.logger.info("\t".join(l))
            #if total > 100:
            #    break
            total += 1
            time.sleep(1)
            loginfo.logger.info("total: {}, recall: {}, correct:{}, accuracy: {}%, ambiguity：{}".format(total, recall, correct, correct * 100.0 / recall, ambiguity))
        #loginfo.logger.info("total: {}, recall: {}, correct:{}, accuracy: {}%".format(total, recall, correct, correct*100.0/recall))
    pass


pattern = re.compile(u'[\\s,.<>/?:;\'\"[\\]{}()\\|~!@#$%^&*\\-_=+，。《》、？：；“”‘’｛｝【】（）…￥！—┄－]+')


def estimate_answer(candidate, answer):
    '''
    :param candidate:
    :param answer:
    :return:
    '''
    candidate = candidate.strip().lower()
    answer = answer.strip().lower()
    if candidate == answer:
        return True

    if not answer.isdigit() and candidate.isdigit():
        candidate_temp = "{:.5E}".format(int(candidate))
        if candidate_temp == answer:
            return True
        candidate_temp == "{:.4E}".format(int(candidate))
        if candidate_temp == answer:
            return True

    return False


def kb_fuzzy_sematic_test():
    '''
    进行问答测试：
    1、 实体检索
    2、 属性映射——语义匹配Tencent Word Embedding
    3、 答案组合
    :return:
    '''
    with open(file) as f:
        total = 0
        recall = 0
        correct = 0
        ambiguity = 0    # 属性匹配正确但是答案不正确

        for line in f:
            try:
                total += 1

                question, entity, attribute, answer, ner = line.split("\t")
                ner = ner.replace("#", "").replace("[UNK]", "%").replace("\n", "")
                # case: entity Fuzzy Match
                sql_e0_a1 = "select * from ods_logs.ods_profile_kb_temp where entity like '%" + ner + "%' order by length(entity) asc limit 20"
                result_e0_a1 = load_data_kudu.load_data(sql_e0_a1, True)

                if len(result_e0_a1) > 0:
                    recall += 1

                    flag_fuzzy = True
                    # 非语义匹配，加快速度
                    flag_ambiguity = True
                    for l in result_e0_a1:
                        if l[1] in question or l[1].lower() in question or l[1].upper() in question:
                            flag_fuzzy = False

                            if estimate_answer(l[2], answer):
                                correct += 1
                                flag_ambiguity = False
                            else:
                                loginfo.logger.info("\t".join(l))

                    # 非语义匹配成功，继续下一次
                    if not flag_fuzzy:

                        if flag_ambiguity:
                            ambiguity += 1

                        time.sleep(1)
                        loginfo.logger.info("total: {}, recall: {}, correct:{}, accuracy: {}%, ambiguity：{}".format(total, recall, correct, correct * 100.0 / recall, ambiguity))
                        continue

                    # 语义匹配
                    result = get_word_list_str(question)
                    result_split_stop = [w for w in result if w not in stoplist and w not in ner and len(pattern.sub(u'', w)) > 0]
                    #result_split_stop = ['名字', '是', '什么']
                    loginfo.logger.info(result_split_stop)
                    result_e0_a1 = load_data_kudu.load_data(sql_e0_a1, True)
                    result_df = pd.DataFrame(result_e0_a1, columns=['entity', 'attribute', 'value'])
                    loginfo.logger.info(result_df.head(100))

                    if len(result_split_stop) < 1:
                        loginfo.logger.info("cannot fuzzy match entity: " + sql_e0_a1)
                        continue
                    else:
                        question_word_vec = get_wrod_vector(",".join(result_split_stop))
                        #loginfo.logger.info(question_word_vec)
                        attribute_list = list(set(result_df["attribute"].tolist()))
                        try:
                            params = urllib.parse.urlencode({'word': ",".join(attribute_list)})
                            # loginfo.logger.info("url: http://192.168.9.91:8011/tencent_model?%s" % params)
                            f = urllib.request.urlopen("http://192.168.9.91:8011/tencent_model?%s" % params, timeout=60).read()
                            attribute_vec = eval(f.decode("utf-8"))
                            if attribute_vec['iRet'] != 0:
                                loginfo.logger.info("cannot get the attribute vector")
                                continue
                        except Exception as e:
                            loginfo.logger.info("error attribute vector %s" % repr(e))


                        question_candicate_sim = [(k, cosine_similarity(v, question_word_vec)) for (k, v) in attribute_vec.items() if k not in ("iRet", "desc")]
                        question_candicate_sort = sorted(question_candicate_sim, key=lambda candicate: candicate[1], reverse=True)
                        loginfo.logger.info("\n".join([str(k)+" "+str(v) for (k, v) in question_candicate_sort]))

                        answer_candicate_df = result_df[result_df["attribute"] == question_candicate_sort[0][0]]
                        for row in answer_candicate_df.index:
                            if estimate_answer(answer_candicate_df.loc[row, "value"], answer):
                                correct += 1
                            else:
                                loginfo.logger.info("\t".join(answer_candicate_df.loc[row].tolist()))
                time.sleep(1)
                loginfo.logger.info("total: {}, recall: {}, correct:{}, accuracy: {}%, ambiguity：{}".format(total, recall, correct, correct * 100.0 / recall, ambiguity))
            except Exception as e:
                loginfo.logger.info("the question id % d occur error %s" % (total, repr(e)))


def kb_fuzzy_classify_test():
    '''
    进行问答测试：
    1、 实体检索
    2、 属性映射——bert分类/文本相似度
    3、 答案组合
    :return:
    '''
    with open(file) as f:
        total = 0
        recall = 0
        correct = 0
        ambiguity = 0    # 属性匹配正确但是答案不正确

        for line in f:
            try:
                total += 1

                question, entity, attribute, answer, ner = line.split("\t")
                ner = ner.replace("#", "").replace("[UNK]", "%").replace("\n", "")
                # case: entity Fuzzy Match
                sql_e0_a1 = "select * from ods_logs.ods_profile_kb_temp where entity like '%" + ner + "%' order by length(entity) asc limit 20"
                result_e0_a1 = load_data_kudu.load_data(sql_e0_a1, True)

                if len(result_e0_a1) > 0:
                    recall += 1

                    flag_fuzzy = True
                    # 非语义匹配，加快速度
                    flag_ambiguity = True
                    for l in result_e0_a1:
                        if l[1] in question or l[1].lower() in question or l[1].upper() in question:
                            flag_fuzzy = False

                            if estimate_answer(l[2], answer):
                                correct += 1
                                flag_ambiguity = False
                            else:
                                loginfo.logger.info("\t".join(l))

                    # 非语义匹配成功，继续下一次
                    if not flag_fuzzy:

                        if flag_ambiguity:
                            ambiguity += 1

                        time.sleep(1)
                        loginfo.logger.info("total: {}, recall: {}, correct:{}, accuracy: {}%, ambiguity：{}".format(total, recall, correct, correct * 100.0 / recall, ambiguity))
                        continue

                    # 语义匹配
                    result_df = pd.DataFrame(result_e0_a1, columns=['entity', 'attribute', 'value'])
                    loginfo.logger.info(result_df.head(100))

                    attribute_candicate_sim = [(k, bs.predict(question, k)[0][1]) for k in result_df['attribute'].tolist()]
                    attribute_candicate_sort = sorted(attribute_candicate_sim, key=lambda candicate: candicate[1], reverse=True)
                    loginfo.logger.info("\n".join([str(k)+" "+str(v) for (k, v) in attribute_candicate_sort]))

                    answer_candicate_df = result_df[result_df["attribute"] == attribute_candicate_sort[0][0]]
                    for row in answer_candicate_df.index:
                        if estimate_answer(answer_candicate_df.loc[row, "value"], answer):
                            correct += 1
                        else:
                            loginfo.logger.info("\t".join(answer_candicate_df.loc[row].tolist()))
                time.sleep(1)
                loginfo.logger.info("total: {}, recall: {}, correct:{}, accuracy: {}%, ambiguity：{}".format(total, recall, correct, correct * 100.0 / recall, ambiguity))
            except Exception as e:
                loginfo.logger.info("the question id % d occur error %s" % (total, repr(e)))


if __name__ == '__main__':
    #embedding_index = load_embedding('/home/mqq/zwshi/ner_model/Tencent_AILab_ChineseEmbedding.txt')
    kb_fuzzy_classify_test()