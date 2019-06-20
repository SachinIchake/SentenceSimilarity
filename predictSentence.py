from sentenceSimilaryService import semtenceSimilarity


def predictAnswer(kb_id, userQuery, top_n):
    # DBUtil.getData('kb_qna',kb_id)
    sc = semtenceSimilarity()
    sc.loadTrainingData()
    matchingAnswer = sc.processSentence(userQuery, top_n)
    print(matchingAnswer)


if __name__ == "__main__":
    userQuery = 'What is ICC Monitoring?'
    predictAnswer({'kb_id': 1}, userQuery, 20)
