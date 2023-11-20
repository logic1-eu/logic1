import datetime
import logging
import time


class DeltaTimeFormatter(logging.Formatter):

    time_since_start_time = time.time() - logging._startTime  # type: ignore

    def format(self, record):
        timestamp = record.relativeCreated / 1000 - self.time_since_start_time
        utc = datetime.datetime.utcfromtimestamp(timestamp)
        record.delta = utc.strftime("%H:%M:%S,%f")
        return super().format(record)

    def reset_clock(self):
        self.time_since_start_time = time.time() - logging._startTime  # type: ignore


class TimePeriodFilter(logging.Filter):

    active = False
    last_log = 0
    rate = 0

    def filter(self, record):
        if not self.active:
            return True
        now = time.time()
        if now - self.last_log >= self.rate:
            self.last_log = now
            return True
        return False

    def off(self):
        self.active = False

    def on(self):
        self.active = True

    def set_rate(self, rate):
        self.rate = rate
