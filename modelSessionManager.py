import global_variables


class SessionManager:
    @classmethod
    def initialiseSession(self):
        pass

    @classmethod
    def getModelSessionList(self):
        return global_variables.modelSessionList

    @classmethod
    def getModelSessionByKbid(self, kbid):
        modelSessions = self.getModelSessionList()
        modelSession = modelSessions.get(kbid, {})
        return modelSession

    @classmethod
    def deleteModelSessionByKbid(self,kbid):
         modelSession = self.getModelSessionList()
         try:
             del modelSession[kbid]
         except Exception as ex :
             pass


    @classmethod
    def setModelSessionByKbid(self, kbid, model_session=None):
        try:
            modelSession = self.getModelSessionList()
            if kbid not in modelSession:
                modelSession[kbid] = {}

            modelSession[kbid] = model_session
            # return modelSession
        except Exception as ex:
            print(ex)
            raise ex
