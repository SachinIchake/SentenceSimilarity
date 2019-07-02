import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

import global_variables
from config import constants
from databaseUtils import DBUtil
from modelSessionManager import SessionManager


class SentenceSemanticService:

    @classmethod
    def embed_customer_data(cls, customer_all_statement):
        sts_encode1 = tf.nn.l2_normalize(global_variables.sentence_embed(customer_all_statement))
        sts_encode1 = global_variables.kbid_session.run(sts_encode1)
        return sts_encode1

    @classmethod
    def define_placeholders(cls):

        global_variables.kbid_session = tf.Session()

        global_variables.sentence_embed = hub.Module(constants.ENV["TFHUB_SENTENCE_MODEL_DIR"])
        global_variables.sts_encode11 = tf.placeholder(tf.float32, shape=(None, 512))
        global_variables.sts_encode3 = tf.placeholder(tf.float32, shape=(None, 512))
        global_variables.sts_input2 = tf.placeholder(tf.string, shape=(None))
        global_variables.sts_encode2 = global_variables.sentence_embed(global_variables.sts_input2)
        global_variables.sim_scores = tf.reduce_sum(
            tf.multiply(global_variables.sts_encode11, tf.nn.l2_normalize(global_variables.sts_encode3)), axis=1)
        with global_variables.kbid_session.as_default() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())

    @classmethod
    def create_only_statements_list(cls, unique_statement_list):
        only_statements = []
        for item in unique_statement_list:
            only_statements.append(item[0])
        return only_statements

    @classmethod
    def process(cls, userQuery, kbid, topN):
        customer_session = cls.getModelSession(kbid)
        unique_data_with_model = customer_session["unique_data_with_model"]
        customer_embed = customer_session["customer_embed"]
        matched_statement = cls.getScore(userQuery, unique_data_with_model, customer_embed, topN)

        return matched_statement

    @classmethod
    def getModelSession(cls, kbid):
        customerModelSession = SessionManager.getModelSessionByKbid(kbid)
        return customerModelSession

    @classmethod
    def restoreModels(cls, customerList):
        try:
            cls.initilize_placeholders()
            cls.loadTrainedDataStoreSession(customerList)
        except Exception as e:
            print('Confidence Checker Model Not Restored for All Customer : ' + str(e))

    @classmethod
    def loadTrainedDataStoreSession(cls, kbid):
        is_model_loaded = SessionManager.getModelSessionByKbid(kbid)

        if not is_model_loaded:
            try:
                unique_data_list = DBUtil.getData('kb_qna', {'kb_id': kbid})
                unique_statement_list = cls.create_only_statements_list(unique_data_list)
                customer_embed = cls.embed_customer_data(unique_statement_list)
                embeding_dict = {"unique_data_with_model": unique_data_list,
                                 "unique_statement_list": unique_statement_list,
                                 "customer_embed": customer_embed}
                SessionManager.setModelSessionByKbid(kbid=kbid, model_session=embeding_dict)
                print('Confidence Checker Model Restored for Customer : ' + str(kbid))

            except Exception as e:
                print('Confidence Checker Model Not Restored for Customer : ' + str(e))

    @classmethod
    def column(self, matrix, i):
        return [row[i] for row in matrix]

    @classmethod
    def run_sts_benchmark(cls, statement_list, text_a, userQuery, topN):
        """Returns the similarity scores"""
        sess = global_variables.kbid_session
        emebb = sess.run([global_variables.sts_encode2], feed_dict={global_variables.sts_input2: userQuery})
        embedding_uq_list = []
        for i in range(len(statement_list)):
            embedding_uq_list.append(emebb[0])
        embedding_uq_list = np.squeeze(embedding_uq_list, axis=1)
        scores = sess.run(
            [global_variables.sim_scores],
            feed_dict={
                global_variables.sts_encode11: text_a,
                global_variables.sts_encode3: embedding_uq_list
            })

        Confidence_Score = (scores[0] * len(statement_list)) * 100.

        uniqueData = np.array(statement_list)

        matchingAnswer = sorted(
            list(zip(SentenceSemanticService.column(uniqueData, 0), Confidence_Score,
                     SentenceSemanticService.column(uniqueData, 2),
                     SentenceSemanticService.column(uniqueData, 1))),
            key=lambda x: x[1], reverse=True)[0:topN]
        return matchingAnswer

    @classmethod
    def getScore(cls, userQuery, statement_list, customer_embed, topN):
        matchingsentence= None
        try:
            line_c = [userQuery.lower()]
            matchingsentence = cls.run_sts_benchmark(statement_list, customer_embed,
                                                     line_c,
                                                     topN)
        except Exception as e:
            print(e)

        return matchingsentence
