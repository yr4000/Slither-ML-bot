#### this is a tester file, it shows how to use log_utils.py####
from log_utils import *

L = Logger()
L.mylogger_spacer()
L.mylogger ('write me down', type='info')
L.mylogger ('write me d', type='error')
L.mylogger ('write me do', type='warning')