
import pymongo as pymongo

from  config.constants import ENV

class DBUtil:
    connection = None

    def __init__(self):
        pass

    @classmethod
    def getConnection(cls):
        try:
            if cls.connection is not None:
                return cls.connection



            dbUrl = ENV["MONGO_HOST"]
            dbSchema = ENV["MONGO_DB"]
            mongoClient = pymongo.MongoClient(dbUrl)
            cls.connection = mongoClient[dbSchema]

        except Exception as e:
            print(e)
        return cls.connection

    @classmethod
    def getData(cls, tableName, filterDict={}, projection=[], pageNumber=1, pageSize=100000,
                sort=['_id', pymongo.ASCENDING]):
        try:
            connection = cls.getConnection()
            skipValue = (pageNumber - 1) * pageSize
            limitValue = pageSize
            response = connection[tableName].find(filterDict).skip(skipValue).limit(limitValue).sort(*sort)
            with open(ENV["TRAINING_DATA"],'w') as fp:
                for doc in response:
                    dataToWrite = doc['question'] + '\t' + doc['answer'] + '\n'
                    fp.write(dataToWrite)

        except Exception as e:
            print("Exception in getData for " + tableName)
            print(e)
        return []

    @classmethod
    def upsert(cls, tableName, filter={}, data={}):
        try:
            connection = cls.getConnection()
            response = connection[tableName].update(filter, {
                '$set': data
            }, upsert=False, multi=True)
            return list(response)
        except Exception as e:
            print("Exception in getData for " + tableName)
            print(e)



def main():
    # DBUtil.getConnection()
    DBUtil.getData('kb_qna',{'kb_id':1})

if __name__ == '__main__':
    main()
