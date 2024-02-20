""" Import this module to raise an exception when SIGINT is received. """

import signal

class KilledException(Exception):
    pass

def exit_gracefully(signum, frame):
    raise KilledException()

# Bind function call to SIGINT signal
signal.signal(signal.SIGINT, exit_gracefully)

print('SIGINT successfully bind to an exception.', flush=True)