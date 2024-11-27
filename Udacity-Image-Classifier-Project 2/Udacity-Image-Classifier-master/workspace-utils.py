######
import signal
from contextlib import contextmanager
import requests

# Constants for time intervals
DEFAULT_DELAY = DEFAULT_INTERVAL = 4 * 60  # time delay in seconds
MIN_DELAY = MIN_INTERVAL = 2 * 60  # minimum allowed time intervals
KEEP_ALIVE_API_URL = "https://nebula.udacity.com/api/v1/remote/keep-alive"
METADATA_URL = "http://metadata.google.internal/computeMetadata/v1/instance/attributes/keep_alive_token"
METADATA_HEADERS = {"Metadata-Flavor": "Google"}

def request_signal_handler(headers):
    """
    Creates a handler to send POST requests at the specified intervals.
    """
    def signal_handler(signum, frame):
        requests.post(KEEP_ALIVE_API_URL, headers=headers)
    return signal_handler

@contextmanager
def maintain_active_session(timeout=DEFAULT_DELAY, interval=DEFAULT_INTERVAL):
    """
    Keeps the session active by sending periodic "keep-alive" requests.

    Example usage:

    from workspace_utils import maintain_active_session

    with maintain_active_session():
        # Perform long-running tasks here
    """
    token = requests.get(METADATA_URL, headers=METADATA_HEADERS).text
    headers = {'Authorization': "STAR " + token}
    
    timeout = max(timeout, MIN_DELAY)
    interval = max(interval, MIN_INTERVAL)
    
    previous_handler = signal.getsignal(signal.SIGALRM)
    
    try:
        signal.signal(signal.SIGALRM, request_signal_handler(headers))
        signal.setitimer(signal.ITIMER_REAL, timeout, interval)
        yield
    finally:
        signal.signal(signal.SIGALRM, previous_handler)
        signal.setitimer(signal.ITIMER_REAL, 0)

def keep_process_alive(iterable, timeout=DEFAULT_DELAY, interval=DEFAULT_INTERVAL):
    """
    Keeps a long-running process active while iterating over a given iterable.

    Example usage:

    from workspace_utils import keep_process_alive

    for item in keep_process_alive(range(5)):
        # Perform heavy processing on each item here
    """
    with maintain_active_session(timeout, interval):
        yield from iterable
