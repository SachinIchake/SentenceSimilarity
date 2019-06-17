import os

import gridfs
import pymongo as pymongo



class DBUtil:
    connection = None

    def __init__(self):
        pass

    @classmethod
    def getConnection(cls):
        try:
            if cls.connection is not None:
                return cls.connection


            import util.Constants as CommonConstants            
            dbUrl = CommonConstants.ENV["DB_URL_BOT"]
            dbSchema = 'QNASchema'
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
            return list(response)
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

    @classmethod
    def insert(cls, tableName, data={}):
        try:
            connection = cls.getConnection()
            response = connection[tableName].insert(data)
            return response
        except Exception as e:
            print("Exception in getData for " + tableName)
            print(e)

    @classmethod
    def storeFile(cls, filePath, fileName):
        connection = cls.getConnection()
        fs = gridfs.GridFS(connection, collection="TFModelStorage")
        fileId = fs.put(open(filePath, 'rb'), filename=fileName)
        return fileId
