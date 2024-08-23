import datetime
import logging
import time


class DeltaTimeFormatter(logging.Formatter):
    """Allows to log the time relative to a reference time by adding an
    attribute `delta` to the :class:`.logging.LogRecord`.

    >>> import logging, sys, time
    >>> logger = logging.getLogger('demo')
    >>> stream_handler = logging.StreamHandler(stream=sys.stdout)
    >>> delta_time_formatter = DeltaTimeFormatter('%(delta)s: %(message)s')
    >>> stream_handler.setFormatter(delta_time_formatter)
    >>> logger.addHandler(stream_handler)
    >>> delta_time_formatter.set_reference_time(time.time())
    >>> time.sleep(0.01)
    >>> logger.warning('Hello world!')  # doctest: +SKIP
    0:00:00.012: Hello world!
    """

    _time_since_start_time = time.time() - logging._startTime  # type: ignore

    def format(self, record: logging.LogRecord) -> str:
        timestamp = record.relativeCreated / 1000 - self._time_since_start_time
        delta = datetime.timedelta(seconds=timestamp)
        record.delta = str(delta)[:-3]
        return super().format(record)

    def get_reference_time(self) -> float:
        """Get the reference time in seconds since the :ref:`epoch <epoch>`.
        This is compatible with the output of :func:`.time.time`.
        """
        return self._time_since_start_time + logging._startTime  # type: ignore

    def set_reference_time(self, reference_time: float) -> None:
        """Set the reference time to `reference_time` seconds since the
        :ref:`epoch <epoch>`. This specification of `reference_time` is
        compatible with the output of :func:`.time.time`.
        """
        self._time_since_start_time = reference_time - logging._startTime  # type: ignore


class RateFilter(logging.Filter):
    """Allows to specify a log rate, which specifies the minimal time in
    seconds that has to pass between two logs. The initial value of the
    rate ist 0.0. The filter is initially on.

    >>> import logging, sys
    >>> logger = logging.getLogger('demo')
    >>> stream_handler = logging.StreamHandler(stream=sys.stdout)
    >>> logger.addHandler(stream_handler)
    >>> rate_filter = RateFilter()
    >>> rate_filter.set_rate(0.001)
    >>> logger.addFilter(rate_filter)
    >>> for count in range(1000):
    ...     logger.warning(f'{count=}')  # doctest: +SKIP
    count=0
    count=276
    count=571
    count=868
    """

    def __init__(self) -> None:
        self.active = True
        self.last_log = 0.0
        self.rate = 0.0

    def filter(self, record: logging.LogRecord) -> bool:
        if not self.active:
            return True
        now = time.time()
        if now - self.last_log >= self.rate:
            self.last_log = now
            return True
        return False

    def off(self) -> None:
        """Turn filter off.
        """
        self.active = False

    def on(self) -> None:
        """Turn filter on.
        """
        self.active = True

    def set_rate(self, rate: float) -> None:
        """Set the log rate to `rate` seconds.
        """
        self.rate = rate


class Timer:
    """A simple timer measuring the wall time in seconds relative to the last
    :meth:`.reset`. Instances of the Timer are implicitly reset when they are
    created.

    >>> import time
    >>> timer = Timer()
    >>> timer.get()  # doctest: +SKIP
    1.7881393432617188e-05
    >>> time.sleep(0.1)
    >>> timer.get()  # doctest: +SKIP
    0.10515975952148438
    """

    def __init__(self) -> None:
        """Reset new instance.
        """
        self.reset()

    def get(self) -> float:
        """Get the wall time since last :meth:`.reset` in seconds.
        """
        return time.time() - self._reference_time

    def reset(self) -> None:
        """Reset the timer to 0.0 seconds.
        """
        self._reference_time = time.time()
