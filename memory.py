class MemoryStore:
    def __init__(self):
        self.data = {}

    def write(self, key: str, value: str):
        self.data[key] = value

    def read(self, key: str):
        return self.data.get(key, None)

    def all(self):
        return self.data
        

memory = MemoryStore()
