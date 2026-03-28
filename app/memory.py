class Memory:
    def __init__(self):
        self.data = []

    def add(self, q, a):
        self.data.append((q, a))

    def get(self):
        return "\n".join([f"Q:{q} A:{a}" for q, a in self.data])