import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        
    def forward(self, x):
        return torch.sigmoid(self.linear(x))
    
    def fit(self, X_train, y_train, num_epochs=10000, lr=0.01):
        criterion = nn.BCELoss()
        optimizer = optim.SGD(self.parameters(), lr=lr)

        for epoch in range(num_epochs):
            # Forward pass
            outputs = self(X_train)
            loss = criterion(outputs.squeeze(), y_train)
    
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            if (epoch+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    def predict(self, X_test):
        with torch.no_grad():
            self.eval()
            outputs = self(X_test)
            predicted = (outputs >= 0.5).squeeze().int()
            return predicted

if __name__ == "__main__":

    # Step 1: Prepare the data
    # Generate some dummy data
    np.random.seed(0)
    X_train = np.random.randn(100, 2)  # 100 samples with 2 features each
    y_train = np.random.randint(2, size=100)  # Binary labels (0 or 1)
    X_test = np.random.randn(20, 2)
    y_test = np.random.randint(2, size=20)

    # Convert the numpy arrays to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Create an instance of the LogisticRegression class
    input_size = 2
    model = LogisticRegression(input_size)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test data
    predicted = model.predict(X_test)

    # Evaluate the model's performance
    accuracy = (predicted == y_test).sum().item() / y_test.size(0)
    print(f'Test Accuracy: {accuracy:.2f}')
