from functools import wraps
import logging


def get_execution_time(func):
    import time

    @wraps(func)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time() - t1
        print(f'This job took {t2} seconds')
        return result

    return wrapper


def catch_and_log_errors(orig_func):

    logger = logging.basicConfig(filename=f'{orig_func.__name__}.log', level=logging.DEBUG)

    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        try:
            results = orig_func(*args, **kwargs)
        except Exception as e:
            print(f'ERROR OCCURRED: {e}')
            logging.error(f"Got Exception {e}")
            logging.exception(e)
            results = None
        return results

    return wrapper
    pass
