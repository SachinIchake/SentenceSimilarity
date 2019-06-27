from flask import Flask
from flask import request, json
from flask import jsonify
import all_global_var
from SentenceSemanticService import ConfidenceCheckerService
from modelSessionManager import SessionManager

app = Flask(__name__)


@app.before_first_request
def before_request():
    try:
        all_global_var.modelSessionList = {}
        all_global_var.all_kbids = {}


    except Exception as e:
        raise (e)

@app.route('/reload', methods=['GET', 'POST'])
def runReload():
    try:
        kbid = json.loads(request.data)['kbid']
        print(kbid)

        ConfidenceCheckerService.initilize_placeholders()


        ConfidenceCheckerService.loadTrainedDataConfidence(kbid)
        return  jsonify({"Result":"Model loaded succefully"})
    except Exception as e:
        return  jsonify({"Result":"Error while loading the model."})





@app.route('/predict', methods=['GET', 'POST'])
def runPrediction():
    try:
        userQuery = json.loads(request.data)['query']
        kbid = json.loads(request.data)['kbid']
        topN = json.loads(request.data)['TopN']
        print(userQuery)
        print(kbid)

        if not SessionManager.getModelSessionByKbid(kbid):
            ConfidenceCheckerService.initilize_placeholders()


        ConfidenceCheckerService.loadTrainedDataConfidence(kbid)
        # userQuery = 'How many online surveys can I participate'
        matched_statement = ConfidenceCheckerService.process(userQuery=userQuery, kbid=kbid, topN=topN)
        print(matched_statement)

        result = []
        for match_question,matched_score,statement,answer  in matched_statement:
            result.append({ "matched_statement": match_question, "score": str(matched_score),"statement": statement,"matched_answer": answer})

        return jsonify(result)


    except Exception as e:
        print(e)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=8080)
