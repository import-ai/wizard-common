import asyncio
import time
from collections import deque


class TPSCalculator:
    def __init__(self, window_seconds: int = 10):
        self.window_seconds = window_seconds
        self.lock = asyncio.Lock()
        self.queue: deque[float] = deque()
        self.total = 0

    async def count(self) -> None:
        async with self.lock:
            self.total += 1
            self.queue.append(time.time())

    async def tps(self) -> float:
        async with self.lock:
            cut_off = time.time() - self.window_seconds
            while self.queue and self.queue[0] < cut_off:
                self.queue.popleft()
            if (count := len(self.queue)) > 1 and self.queue[-1] > self.queue[0]:
                return round(count / (self.queue[-1] - self.queue[0]), 2)
            return 0.0
