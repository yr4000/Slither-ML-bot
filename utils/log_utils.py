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


'''
#This is  tester.
if __name__ == '__main__':
	L = Logger('yair')
	L.write_spacer()
	L.write_to_log('write me down', type='info')
	L.write_to_log('write me d', type='error')
	L.write_to_log('write me do', type='warning')
'''


###!!!CODE FROM HERE ON IS FOR DEBUGGING PURPOSES ONLY!!!###	
#LOGGER.mylogger('my message is da best')
#mylogger_spacer()
#LOGGER.mylogger('my message is da best')
#mylogger_spacer()
#LOGGER.mylogger('my message is da best')
#mylogger_spacer()
#LOGGER.mylogger('my message is da best')
#mylogger_spacer()

#for i in range(10):
#	LOGGER.mylogger('love is in the air')
#	if (i%2):
#		LOGGER.mylogger('bytwo', type='warning')
#	elif (i%3):
#		LOGGER.mylogger('bythree', type='error')
#	elif (i%5):
#		LOGGER.mylogger('byfive', type='infffo')