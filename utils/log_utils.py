import logging
LOG_FILENAME = 'ML.log'
SPACER = '======================================================================================================================='

class Logger:
	def __init__(self):
		return
	def mylogger (self, message, type='info'):
		logging.basicConfig(format='%(asctime)s-%(levelname)s: %(message)s', filename=LOG_FILENAME, level=logging.DEBUG)
		if (type == 'info'):
			logging.info(message)
		elif (type == 'warning'):
			logging.warning(message)
		elif (type == 'error'):
			logging.error(message)
		else:
			logging.info('unkown type: ' + type + ": " + message)

	def mylogger_spacer (self):
		logging.basicConfig(format='%(asctime)s-%(levelname)s: %(message)s', filename=LOG_FILENAME, level=logging.DEBUG)
		logging.info("\n" + SPACER)




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