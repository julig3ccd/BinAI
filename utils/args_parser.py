import argparse

class Parser:
    def __init__(self):
        self.parser = argparse.ArgumentParser() 
        #TODO put actual args here
        self.parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
        self.parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
        self.parser.add_argument("--model", type=str, help="Model name")
    
    def get_args(self):
       return self.parser.parse_args()
    