"""Small speed-adaptive filters for noisy pointer coordinates."""
import math


class OneEuroFilter:
    """One Euro low-pass filter: smooths stillness and follows quick motion."""

    def __init__(self, min_cutoff=0.4, beta=0.06, derivative_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.derivative_cutoff = derivative_cutoff
        self.previous = None
        self.previous_derivative = 0.0
        self.previous_time = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def update(self, value, timestamp):
        if not math.isfinite(value):
            return self.previous if self.previous is not None else 0.5
        if self.previous_time is None:
            self.previous = value
            self.previous_time = timestamp
            return value
        dt = timestamp - self.previous_time
        if dt <= 0:
            return self.previous
        derivative = (value - self.previous) / dt
        derivative_alpha = self._alpha(self.derivative_cutoff, dt)
        filtered_derivative = derivative_alpha * derivative + (1.0 - derivative_alpha) * self.previous_derivative
        cutoff = self.min_cutoff + self.beta * abs(filtered_derivative)
        alpha = self._alpha(cutoff, dt)
        filtered = alpha * value + (1.0 - alpha) * self.previous
        self.previous = filtered
        self.previous_derivative = filtered_derivative
        self.previous_time = timestamp
        return filtered

    def reset(self):
        self.previous = None
        self.previous_derivative = 0.0
        self.previous_time = None


class OneEuroFilter2D:
    def __init__(self, min_cutoff=0.4, beta=0.06):
        self.x = OneEuroFilter(min_cutoff, beta)
        self.y = OneEuroFilter(min_cutoff, beta)

    def update(self, point, timestamp):
        return self.x.update(float(point[0]), timestamp), self.y.update(float(point[1]), timestamp)

    def reset(self):
        self.x.reset()
        self.y.reset()
