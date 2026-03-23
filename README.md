# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

It consists of an input layer with 1 neuron, two hidden layers with 4 neurons each, and an output layer with 1 neuron. Each neuron in one layer is connected to all neurons in the next layer, allowing the model to learn complex patterns. The hidden layers use activation functions such as ReLU to introduce non-linearity, enabling the network to capture intricate relationships within the data. 
During training, the model adjusts its weights and biases using optimization techniques like RMSprop or Adam, minimizing a loss function such as Mean Squared Error for regression.The forward propagation process involves computing weighted sums, applying activation functions, and passing the transformed data through layer.

## Neural Network Model
<img width="1060" height="758" alt="image" src="https://github.com/user-attachments/assets/4829d4a4-6e3e-4bb1-95ea-e8f972a2568a" />



## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:Aishwariya s
### Register Number:212224240005
```

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import matplotlib.pyplot as plt

dataset1 = pd.read_csv('/StudentsPerformance[1].csv')

# Encode 'gender' and 'lunch' to numerical values
le_gender = LabelEncoder()
dataset1['gender_encoded'] = le_gender.fit_transform(dataset1['gender'])

le_lunch = LabelEncoder()
dataset1['lunch_encoded'] = le_lunch.fit_transform(dataset1['lunch'])

X = dataset1[['gender_encoded']].values
y = dataset1[['lunch_encoded']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 10)
        self.fc3 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

print("Name: AISHWARIYA S")
print("Reg No: 212224240005")
aishu = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(aishu.parameters(), lr=0.001)

def train_model(aishu, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(aishu(X_train), y_train)
        loss.backward()
        optimizer.step()

        # Corrected typo from yuvas.history to aishu.history
        aishu.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

train_model(aishu, X_train_tensor, y_train_tensor, criterion, optimizer)

with torch.no_grad():
    test_loss = criterion(aishu(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')

loss_df = pd.DataFrame(aishu.history)

loss_df.plot()
print("Name: AISHWARIYA S")
print("Reg No: 212224240005")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()


```
## Dataset Information

<img width="323" height="703" alt="image" src="https://github.com/user-attachments/assets/50280e42-66db-4a67-82b2-b97f5a757fc5" />

## OUTPUT
<img width="383" height="311" alt="image" src="https://github.com/user-attachments/assets/3b0bda52-7380-49d2-bc49-14fea84016e0" />

### Training Loss Vs Iteration Plot

<img width="828" height="616" alt="image" src="https://github.com/user-attachments/assets/28253fb5-da48-45eb-bcf7-2fb7890e41a6" />



## RESULT
The program to develop a neural network regression model for the given dataset has been executed successively
