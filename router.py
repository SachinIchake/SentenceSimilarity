from flask import Flask
from flask import request, json

import all_global_var
from SentenceSemanticService import ConfidenceCheckerService
from modelSessionManager import SessionManager

app = Flask(__name__)


@app.before_first_request
def before_request():
    try:
        all_global_var.modelSessionList = {}


    except Exception as e:
        raise (e)


@app.route('/predict', methods=['GET', 'POST'])
def runPrediction():
    try:
        userQuery = json.loads(request.data)['query']
        kbid = json.loads(request.data)['kbid']
        print(userQuery)
        print(kbid)

        if not SessionManager.getModelSessionByKbid(kbid):
            ConfidenceCheckerService.initilize_placeholders()

        ConfidenceCheckerService.loadTrainedDataConfidence(kbid)
        # userQuery = 'How many online surveys can I participate'
        matched_statement = ConfidenceCheckerService.process(userQuery=userQuery, kbid=kbid, topN=5)
        print(matched_statement)
        return ' '

    except Exception as e:
        print(e)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=8080)
