class Logger():
    def __init__(self, name):
        self.Name = name

    def log(self, text):
        with open(self.Name + ".txt", "a+") as logFile:
            logFile.write(str(text) + "\n")
