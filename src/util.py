import time
import os

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class Logger():
    def __init__(self, path):
        self.logger = open(os.path.join(path, 'log.txt'), 'w')

    def __call__(self, print_string, end='\n'):
        '''print log'''
        print("{}".format(print_string), end=end)
        if end == '\n':
            self.logger.write('{}\n'.format(print_string))
        else:
            self.logger.write('{} '.format(print_string))
        self.logger.flush()

def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))