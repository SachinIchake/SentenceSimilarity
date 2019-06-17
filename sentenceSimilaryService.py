from itertools import repeat

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


# embed = hub.Module(os.environ.get("TFHUB_SENTENCE_MODEL"))


class semtenceSimilarity:

    def __init__(self):
        try:
            self.userQuery = ''
            self.ss_Session = tf.Session()
            # self.sentence_embedding = hub.Module(Constants.ENV["TFHUB_SENTENCE_MODEL_DIR"])
            # self.sentence_embedding = hub.Module('/home/atom/Git/embeddings/tf-model/')
            self.sentence_embedding = hub.Module('/home/atom/Git/embeddings/sentence_model/')
            self.sts_encoding_1 = tf.placeholder(tf.float32, shape=(None, 512))
            self.sts_encoding_3 = tf.placeholder(tf.float32, shape=(None, 512))

            self.sts_input2 = tf.placeholder(tf.string, shape=(None))
            self.sts_encoding_2 = self.sentence_embedding(self.sts_input2)

            self.sim_scores = tf.reduce_sum(
                tf.multiply(self.sts_encoding_1, tf.nn.l2_normalize(self.sts_encoding_3)), axis=1)
            with self.ss_Session.as_default() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.tables_initializer())
        except Exception as e:
            print("Exception occurred while Confidence checker initilize_placeholders ----> " + str(e))

    def getUniqueData(self):
        self.uniqueData = []
        with open('/home/atom/UST/SentenceSimilarity/data/trainingData.txt') as fp:
            for line in fp.readlines():
                if line != '\n':
                    query = line.split('\t')[0].lstrip()
                    answer = line.split('\t')[1].lstrip()

                    self.uniqueData.append([query,answer])

        # self.uniqueData = list(set(uniqueData))


    def embed_customer_data(self, customer_all_statement):

        sts_encode1 = tf.nn.l2_normalize(self.sentence_embedding(customer_all_statement))
        sts_encode1 = self.ss_Session.run(sts_encode1)
        self.trainingDataEmbd =  sts_encode1

    def loadTrainingData(self):
        self.getUniqueData()
        self.embed_customer_data(self.column(self.uniqueData,0))

    def processSentence(self,userQuery):

        max_score, matched_statement, matching_answer = self.calcualteScore(userQuery, self.uniqueData)

        print("Sentence :" + str(userQuery) + "\t\t" + "Matching Sentence:" + str(
            matched_statement) + "\t\t" + "Matching Answer:" + str(
            matching_answer) + "\t\t" + "Score:" + str(max_score))

        return max_score, matched_statement,matching_answer

    def column(self,matrix, i):
        return [row[i] for row in matrix]

    def calculateConfidence(self, userQuery, text_b):
        """Returns the similarity scores"""
        matchingsentence, sentence,matchingAnswer = None, None,None
        sess = self.ss_Session
        emebb = sess.run([self.sts_encoding_2], feed_dict={self.sts_input2: userQuery})
        embedding_uq_list = []
        for i in range(len(self.uniqueData)):
            embedding_uq_list.append(emebb[0])
        embedding_uq_list = np.squeeze(embedding_uq_list, axis=1)
        scores = sess.run(
            [self.sim_scores],
            feed_dict={
                self.sts_encoding_1: self.trainingDataEmbd,
                self.sts_encoding_3: embedding_uq_list
            })

        Confidence_Score = (scores[0] * len(self.uniqueData)) * 100.

        max_score = max(Confidence_Score)
        uniqueData= np.array(self.uniqueData)
        a = zip(self.column(uniqueData, 0), Confidence_Score, text_b,self.column(uniqueData, 1))
        for match_sentence, Confidence_Score, sentence,matching_answer in a:
            if Confidence_Score == max_score:
                matchingsentence = match_sentence
                matchingAnswer = matching_answer
                break
        return max_score, matchingsentence, matchingAnswer

    def calcualteScore(self, userQuery, statement_list):
        max_score, matchingsentence, matchingAnswer = None, None,None
        try:
            line_c = [userQuery.lower()]
            text_b = [x for item in line_c for x in repeat(item, len(statement_list))]
            max_score, matchingsentence, matchingAnswer= self.calculateConfidence(line_c,text_b)
            return max_score, matchingsentence, matchingAnswer
        except Exception as e:
            print(e)

        return max_score, matchingsentence,matchingAnswer


    def sentenceSimilarityScore(self,userQuery):

        # self.loadTrainingData()
        max_score, matched_statement =self.process(userQuery)

        return max_score, matched_statement

