import datetime
import logging
import time


class DeltaTimeFormatter(logging.Formatter):

    time_since_start_time = time.time() - logging._startTime  # type: ignore

    def format(self, record):
        timestamp = record.relativeCreated / 1000 - self.time_since_start_time
        delta = datetime.timedelta(seconds=timestamp)
        record.delta = str(delta)[:-3]
        return super().format(record)

    def reset_clock(self):
        self.time_since_start_time = time.time() - logging._startTime  # type: ignore


class RateFilter(logging.Filter):

    def __init__(self):
        self.active = True
        self.last_log = 0
        self.rate = 0

    def filter(self, record):
        if not self.active:
            return True
        now = time.time()
        if now - self.last_log >= self.rate:
            self.last_log = now
            return True
        return False

    def is_due_soon(self, tolerance: float = 0) -> bool:
        return time.time() - self.last_log >= self.rate - tolerance

    def off(self):
        self.active = False

    def on(self):
        self.active = True

    def set_rate(self, rate):
        self.rate = rate
