from itertools import repeat

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

import all_global_var
from config import constants
from databaseUtils import DBUtil
from modelSessionManager import SessionManager


class ConfidenceCheckerService:

    @classmethod
    def embed_customer_data(cls, customer_all_statement):
        sts_encode1 = tf.nn.l2_normalize(all_global_var.sentence_embed(customer_all_statement))
        sts_encode1 = all_global_var.checker_session.run(sts_encode1)
        return sts_encode1

    @classmethod
    def initilize_placeholders(cls):
        try:

            all_global_var.checker_session = tf.Session()

            all_global_var.sentence_embed = hub.Module(constants.ENV["TFHUB_SENTENCE_MODEL_DIR"])
            all_global_var.sts_encode11 = tf.placeholder(tf.float32, shape=(None, 512))
            all_global_var.sts_encode3 = tf.placeholder(tf.float32, shape=(None, 512))
            all_global_var.sts_input2 = tf.placeholder(tf.string, shape=(None))
            all_global_var.sts_encode2 = all_global_var.sentence_embed(all_global_var.sts_input2)
            all_global_var.sim_scores = tf.reduce_sum(
                tf.multiply(all_global_var.sts_encode11, tf.nn.l2_normalize(all_global_var.sts_encode3)), axis=1)
            with all_global_var.checker_session.as_default() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.tables_initializer())
        except Exception as e:
            print("Exception occurred while Confidence checker initilize_placeholders ----> " + str(e))
            raise e

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
            cls.loadTrainedDataConfidence(customerList)
        except Exception as e:
            print('Confidence Checker Model Not Restored for All Customer : ' + str(e))
            print("Exception occurred while restoreModels ----> " + str(e))

    @classmethod
    def loadTrainedDataConfidence(cls, kbid):
        is_model_loaded = SessionManager.getModelSessionByKbid(kbid)

        if not is_model_loaded :
            try:
                unique_data_list = DBUtil.getData('kb_qna', {'kb_id': kbid})
                unique_statement_list = cls.create_only_statements_list(unique_data_list)
                customer_embed = cls.embed_customer_data(unique_statement_list)
                data = {"unique_data_with_model": unique_data_list,
                        "unique_statement_list": unique_statement_list,
                        "customer_embed": customer_embed}
                customer_session = SessionManager.setModelSessionByKbid(kbid=kbid, model_session=data)
                print('Confidence Checker Model Restored for Customer : ' + str(kbid))

            except Exception as e:
                print('Confidence Checker Model Not Restored for Customer : ' + str(e))
                print("Exception occurred while loadTrainedDataConfidence ----> " + str(e))

    @classmethod
    def column(self, matrix, i):
        return [row[i] for row in matrix]

    @classmethod
    def run_sts_benchmark(cls, statement_list, text_a, userQuery, text_b, topN):
        """Returns the similarity scores"""
        sess = all_global_var.checker_session
        emebb = sess.run([all_global_var.sts_encode2], feed_dict={all_global_var.sts_input2: userQuery})
        embedding_uq_list = []
        for i in range(len(statement_list)):
            embedding_uq_list.append(emebb[0])
        embedding_uq_list = np.squeeze(embedding_uq_list, axis=1)
        scores = sess.run(
            [all_global_var.sim_scores],
            feed_dict={
                all_global_var.sts_encode11: text_a,
                all_global_var.sts_encode3: embedding_uq_list
            })

        Confidence_Score = (scores[0] * len(statement_list)) * 100.

        uniqueData = np.array(statement_list)

        matchingAnswer = sorted(
            list(zip(ConfidenceCheckerService.column(uniqueData, 0), Confidence_Score, text_b,
                     ConfidenceCheckerService.column(uniqueData, 1))),
            key=lambda x: x[1], reverse=True)[0:topN]
        return matchingAnswer

    @classmethod
    def getScore(cls, userQuery, statement_list, customer_embed, topN):
        max_score, matchingsentence, sentence = None, None, None
        try:
            line_c = [userQuery.lower()]
            text_b = [x for item in line_c for x in repeat(item, len(statement_list))]
            matchingsentence = cls.run_sts_benchmark(statement_list, customer_embed,
                                                     line_c,
                                                     text_b, topN)
        except Exception as e:
            print(e)

        return matchingsentence


if __name__ == "__main__":
    ConfidenceCheckerService.initilize_placeholders()

    while (1):
        # userQuery = input('--> ')
        userQuery = 'Do I have to do the pre screening?'
        kbid = int(input('KBID-->'))
        ConfidenceCheckerService.loadTrainedDataConfidence(kbid=kbid)
        matched_statement = ConfidenceCheckerService.process(userQuery=userQuery, kbid=kbid, topN=5)
        print(matched_statement)
