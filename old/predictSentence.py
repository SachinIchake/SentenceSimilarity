from old.sentenceSimilaryService import semtenceSimilarity


def predictAnswer(kb_id, userQuery, top_n):

    matchingAnswer = sc.processSentence(userQuery, top_n)
    print(matchingAnswer)
    # for id , val in enumerate(matchingAnswer):
    #     print(str(id) + ' ' + list(val))


if __name__ == "__main__":
     # DBUtil.getData('kb_qna',kb_id)
    sc = semtenceSimilarity()
    sc.loadTrainingData()
    # userQuery = 'What is ICC Monitoring?'
    while(1):
        userQuery = input('Enter Query Here: ')
        predictAnswer({'kb_id': 1}, userQuery, 10)
