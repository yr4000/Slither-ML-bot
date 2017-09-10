import logging
import time
import os
LOG_FILENAME = 'ML.log'
SPACER = '======================================================================================================================='

class Logger:
    def __init__(self, file_name = ''):
        directory = './logs/'
        if(not os.path.exists(directory)):
            os.makedirs(directory)

        if(file_name == ''):
            self.file_name = directory + LOG_FILENAME
        else:
            t = time.gmtime()
            self.file_name = directory + file_name + '_' + time.strftime("%d%m%Y-%H%M%S") + '.log'
        return

    def write_to_log (self, message, type='info'):
        logging.basicConfig(format='%(asctime)s-%(levelname)s: %(message)s', filename=self.file_name, level=logging.DEBUG)
        if (type == 'info'):
            logging.info(message)
        elif (type == 'warning'):
            logging.warning(message)
        elif (type == 'error'):
            logging.error(message)
        else:
            logging.info('unkown type: ' + type + ": " + message)

    def write_spacer (self):
        logging.basicConfig(format='%(asctime)s-%(levelname)s: %(message)s', filename=self.file_name, level=logging.DEBUG)
        logging.info("\n" + SPACER)