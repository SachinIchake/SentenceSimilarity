import  numpy as np
from flask import Flask , render_template
from flask import request,json
import  requests


from SentenceSemanticService import ConfidenceCheckerService
app = Flask(__name__)

@app.before_first_request
def before_request():
    try:
        ConfidenceCheckerService.initilize_placeholders()

    except Exception as e :
        raise (e)

@app.route('/predict',methods= ['GET','POST'])
def runPrediction():
    try:
        ConfidenceCheckerService.loadTrainedDataConfidence(kbid=1)
        # while (1):
        #     userQuery = input('--> ')
        userQuery = 'How many online surveys can I participate'
        matched_statement = ConfidenceCheckerService.process(userQuery=userQuery, kbid=1, topN=15)
        print(matched_statement)

    except Exception as e:
        print(e)


if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=False,port=8080)
