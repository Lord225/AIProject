import os
import datetime

LOG_DIR_ROOT = 'logs/'
LOG_DIR = os.path.join(LOG_DIR_ROOT, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
MODELS_DIR = 'models/'