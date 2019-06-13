import configparser
import os


class ConfigProvider:
    config = None

    def __init__(self):
        pass

    @classmethod
    def getConfig(cls):
        if cls.config is None:
            cls.config = cls.readConfigFile("../config/messages.ini")
        return cls.config

    @classmethod
    def readConfigFile(cls, filePath):
        """
        purpose: used to read Messages.ini configuration file
        :return: return config object
        """
        config = configparser.ConfigParser()
        config.read(os.path.join(os.path.abspath(os.path.dirname(__file__)), filePath))
        return config

    @classmethod
    def getNerDirectoryNames(cls, customerId, domain):
        return [
            "../KanverseTrainedModels/models/" + customerId + "/ner/" + domain,
            "../KanverseTrainedModels/data/" + customerId + "/ner/" + domain
        ]

    @classmethod
    def getMNDirectoryNames(cls, customerId, domain):
        return [
            "../KanverseTrainedModels/models/" + customerId + "/mn/" + domain,
            "../KanverseTrainedModels/data/" + customerId + "/mn/" + domain
        ]

    @classmethod
    def getQNADirectoryNames(cls, customerId, domain):
        return [
            "../KanverseTrainedModels/models/" + customerId + "/qna/" + domain,
            "../KanverseTrainedModels/data/" + customerId + "/qna/" + domain
        ]

    @classmethod
    def getNerFileNames(cls, customerId, domain):
        return {
            'filename_words': "../KanverseTrainedModels/models/" + customerId + "/ner/" + domain + "/words.txt",
            'filename_tags': "../KanverseTrainedModels/models/" + customerId + "/ner/" + domain + "/tags.txt",
            'filename_chars': "../KanverseTrainedModels/models/" + customerId + "/ner/" + domain + "/chars.txt",
            'filename_dev': "../KanverseTrainedModels/data/" + customerId + "/ner/" + domain + "/train.txt",
            'filename_test': "../KanverseTrainedModels/data/" + customerId + "/ner/" + domain + "/train.txt",
            'filename_train': "../KanverseTrainedModels/data/" + customerId + "/ner/" + domain + "/train.txt",
            'common_glove': "../KanverseTrainedModels/data/glove.6B.300d.txt",
            'input_glove': "../KanverseTrainedModels/models/" + customerId + "/ner/" + domain + "/glove.6B.300d.trimmed.npz",
            'output_log': "../KanverseTrainedModels/models/" + customerId + "/ner/" + domain + "/log.txt",
            'dir_output': "../KanverseTrainedModels/models/" + customerId + "/ner/" + domain + "/",
            'archive_output': "../KanverseTrainedModels/models/" + customerId + "/ner/" + domain,
            'dir_model': "../KanverseTrainedModels/models/" + customerId + "/ner/" + domain + "/"
        }

    @classmethod
    def getMNFileNames(cls, customerId, domain):
        return {
            'filename_dev': "../KanverseTrainedModels/data/" + customerId + "/mn/" + domain + "/dialog-kanverse-API-calls-dev.txt",
            'filename_test': "../KanverseTrainedModels/data/" + customerId + "/mn/" + domain + "/dialog-kanverse-API-calls-dev.txt",
            'filename_train': "../KanverseTrainedModels/data/" + customerId + "/mn/" + domain + "/dialog-kanverse-API-calls-dev.txt",
            'filename_candidates': "../KanverseTrainedModels/data/" + customerId + "/mn/" + domain + "/dialog-kanverse-candidates.txt",
            'output_log': "../KanverseTrainedModels/models/" + customerId + "/mn/" + domain + "/log.txt",
            'archive_output': "../KanverseTrainedModels/models/" + customerId + "/mn/" + domain,
            'dir_model': "../KanverseTrainedModels/models/" + customerId + "/mn/" + domain + "/",
            'vocab_pickel': "../KanverseTrainedModels/models/" + customerId + "/mn/" + domain + "/vocab_pickel"
        }

    @classmethod
    def getQNAFileNames(cls, customerId, domain):
        return {
            'filename_train': "../KanverseTrainedModels/models/" + customerId + "/qna/" + domain + "/train.txt",
            'filename_labels': "../KanverseTrainedModels/models/" + customerId + "/qna/" + domain + "/labels.txt",
            'filename_candidate': "../KanverseTrainedModels/models/" + customerId + "/qna/" + domain + "/candidate.txt",
            'dir_model': "../KanverseTrainedModels/models/" + customerId + "/qna/" + domain + "/",
            'dir_checkpoint': "../KanverseTrainedModels/models/" + customerId + "/qna/" + domain + "/",
            'dir_output': "../KanverseTrainedModels/models/" + customerId + "/qna/" + domain + "/",
            'archive_output': "../KanverseTrainedModels/models/" + customerId + "/qna/" + domain,
        }
