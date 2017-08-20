#### this is a tester file, it shows how to use log_utils.py####
from utils.log_utils import *

L = Logger('yair')
L.write_spacer()
L.write_to_log ('write me down', type='info')
L.write_to_log ('write me d', type='error')
L.write_to_log ('write me do', type='warning')