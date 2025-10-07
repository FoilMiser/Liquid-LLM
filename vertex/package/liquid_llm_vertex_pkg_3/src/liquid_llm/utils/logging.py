import logging, sys
try:
    from pythonjsonlogger import jsonlogger
except Exception:
    jsonlogger = None

def init_logger(level='INFO'):
    logger = logging.getLogger()
    if logger.handlers:
        return
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    if jsonlogger is not None:
        formatter = jsonlogger.JsonFormatter('%(levelname)s %(name)s %(message)s')
        handler.setFormatter(formatter)
    else:
        formatter = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')
        handler.setFormatter(formatter)
    logger.addHandler(handler)

def get_logger(name):
    return logging.getLogger(name)
