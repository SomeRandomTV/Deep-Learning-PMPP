import csv
import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path 
from tqdm import tqdm 
import torch 
import torch.nn as nn
import torch.optim as optim 

EPOCHS = 200

# token encoding 
TOKEN_ENCODING = {
    "ST": 0.0,
    "open": 0.1,
    "read": 0.3,
    "write": 0.7,
    "close": 0.9,
    "ED": 1.0

}

TOKEN_LABELS = {

    "ST": 0,
    "open": 1,
    "read": 2,
    "write": 3,
    "close": 4,
    "ED": 5
}

LABEL_TO_TOK = {

    0: "ST",
    1: "open",
    2: "read",
    3: "write",
    4: "close",
    5: "ED" 
}

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():    # for testing 
    device = torch.device("mps")
else:
    device = torch.device("cpu")

    

def token_encoder(raw_tokens: pd.DataFrame):
    
    X = np.column_stack([

        np.array(raw_tokens["token1"].map(TOKEN_ENCODING)),
        np.array(raw_tokens["token2"].map(TOKEN_ENCODING))
    ])

    y: np.ndarray = np.array(raw_tokens["next_token"].map(TOKEN_LABELS)) 


    return X, y 



class TokenPred(nn.Module):

    def __init__(self, hidden_nodes: int):

        super().__init__()
        self.fc1 = nn.Linear(2, hidden_nodes)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_nodes, 6)

    def forward(self, x: torch.Tensor):

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

def train(model: TokenPred, X_tr: torch.Tensor, y_tr: torch.Tensor, learning_rate: float, hidden_nodes: int):

    criterion = nn.CrossEntropyLoss()
    optimizer =  optim.Adam(model.parameters(), learning_rate)
    
    model.train()
    for epoch in tqdm(range(EPOCHS), desc=f"Hidden Nodes: {hidden_nodes}"):
        optimizer.zero_grad()
        outputs = model(X_tr)
        loss = criterion(outputs, y_tr)
        loss.backward()
        optimizer.step()


def evaluate(model: TokenPred, X_ev: torch.Tensor, y_ev: torch.Tensor) -> float:
    model.eval()

    with torch.no_grad():
        outputs = model(X_ev)
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == y_ev).float().mean().item()

    return accuracy


def predict(model: TokenPred, t1: str, t2: str):

    model.eval()

    with torch.no_grad():
        x = torch.tensor([[TOKEN_ENCODING[t1], TOKEN_ENCODING[t2]]], dtype=torch.float32).to(device)

        output = model(x)
        pred = torch.argmax(output, dim=1)
    
    return LABEL_TO_TOK[pred.item()]
            


    


def main():

    if (len(sys.argv) != 2):
        print("ERROR: Invalid arguments given")
        exit(-123)


    print("Alejandro Rubio")
    print("R11886363")

    input_tokens_path = Path(sys.argv[1])
 

    input_tokens = pd.read_csv(input_tokens_path)
    X_enc, y_enc = token_encoder(input_tokens)

    # shuffle and split tokens 
    idx = np.random.default_rng(42).permutation(len(X_enc))
    X_enc = X_enc[idx]
    y_enc = y_enc[idx]
    X = torch.tensor(X_enc, dtype=torch.float32).to(device)
    y = torch.tensor(y_enc, dtype=torch.long).to(device)

    split = 70
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"On device: {device}")
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    print("----- Token Encoding -----")
    print(json.dumps(TOKEN_ENCODING, indent=2))


    for n_nodes in [3,9]:
        model = TokenPred(hidden_nodes=n_nodes).to(device)

        train(model, X_train, y_train,0.01, n_nodes)
        
        train_accuracy = evaluate(model, X_train, y_train)
        test_accuracy = evaluate(model, X_test, y_test)
        
        pred_1 = predict(model, "open", "close")
        pred_2 = predict(model, "close", "ED")

        print(f"=========== Results ({n_nodes}) ============")
        print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
        print(f"Testing Accuracy:  {test_accuracy * 100:.2f}%")
        print("---------- Predictions ----------")
        print(f" 'open' 'close' -> '{pred_1}'")
        print(f" 'close' 'ED' -> '{pred_2}'")
        print("==================================")


              



if __name__ == "__main__":
    main()
    


    
    

