import os
import datetime

LOG_DIR_ROOT = 'logs/'
RUN_NAME = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_DIR = os.path.join(LOG_DIR_ROOT, RUN_NAME)
MODELS_DIR = 'models/'