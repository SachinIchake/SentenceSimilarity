import all_global_var


class SessionManager:
    @classmethod
    def initialiseSession(self):
        pass

    # @classmethod
    # def getKbIds(cls, kbid):
    #     if len(all_global_var.modelSessionList) == 0:
    #         return 0
    #     else:
    #         for id in all_global_var.all_kbids:
    #             if id == kbid:
    #                 return 1
    #         return 0

    @classmethod
    def getModelSessionList(self):
        return all_global_var.modelSessionList

    @classmethod
    def getModelSessionByKbid(self, kbid):
        modelSessions = self.getModelSessionList()
        modelSession = modelSessions.get(kbid, {})
        return modelSession

    def deleteModelSessionByKbid(self,kbid):
         modelSession = self.getModelSessionList()
         try:
             del modelSession[kbid]
         except Exception as ex :
             print('model is not loaded for kbid: '+ str(kbid))
    @classmethod
    def setModelSessionByKbid(self, kbid, model_session=None):
        try:
            modelSession = self.getModelSessionList()
            if kbid not in modelSession:
                modelSession[kbid] = {}

            modelSession[kbid] = model_session
            return modelSession
        except Exception as ex:
            print(ex)
            raise ex
