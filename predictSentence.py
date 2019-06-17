from sentenceSimilaryService import  semtenceSimilarity

if __name__ == "__main__":

    # userQuery = 'How do I create snapshots for VDC account?'

    sc = semtenceSimilarity()
    sc.loadTrainingData()
    while(1):
        userQuery = input('-->')
        max_score, matched_statement,matchingAnswer = sc.processSentence(userQuery)
