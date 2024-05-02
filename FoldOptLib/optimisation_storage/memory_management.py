import gc


class MemoryCollection:
    def __init__(self):
        self._memory = []

    def add(self, obj):
        self._memory.append(obj)

    def remove(self, obj):
        self._memory.remove(obj)

    def clear(self):
        self._memory.clear()

    def __del__(self):
        self.clear()
        gc.collect()
