import argparse

class Parser:
    def __init__(self):
        self.parser = argparse.ArgumentParser() 
    
        self.parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
        self.parser.add_argument("--warm_up_epochs", type=int, default=0)
        self.parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
        self.parser.add_argument("--model", type=str, help="Model name")
        self.parser.add_argument("--batch_size", type=int, default=8)
        self.parser.add_argument("--num_heads", type=int, default=8)
        self.parser.add_argument("--num_workers", type=int, default=4)
        self.parser.add_argument("--num_hidden_layers", type=int, default=4)
        self.parser.add_argument("--checkpoint", type=str, default=None)
        self.parser.add_argument("--out_dir", type=str,default="out" )
        self.parser.add_argument("--train_data", type=str, default="data/preprocessed_test_data_train")
        self.parser.add_argument("--val_data", type=str, default="data/preprocessed_test_data_val")
        self.parser.add_argument("--test_data", type=str, default="data/preprocessed_test_data/curl__pools.json")
        self.parser.add_argument("--create_opcode_ids", action="store_true")
        self.parser.add_argument("--masking", type=str, default="MLM")


    def get_args(self):
       return self.parser.parse_args()
    