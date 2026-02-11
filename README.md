# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

It consists of an input layer with 1 neuron, two hidden layers with 4 neurons each, and an output layer with 1 neuron. Each neuron in one layer is connected to all neurons in the next layer, allowing the model to learn complex patterns. The hidden layers use activation functions such as ReLU to introduce non-linearity, enabling the network to capture intricate relationships within the data. 
During training, the model adjusts its weights and biases using optimization techniques like RMSprop or Adam, minimizing a loss function such as Mean Squared Error for regression.The forward propagation process involves computing weighted sums, applying activation functions, and passing the transformed data through layer.

## Neural Network Model

![image](https://github.com/user-attachments/assets/ee9acc10-42da-48f5-9a05-b860601c1f28)


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
```python
class Neuralnet(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1=nn.Linear(1,8)
    self.fc2=nn.Linear(8,10)
    self.fc3=nn.Linear(10,1)
    self.relu=nn.ReLU()
    self.history={'loss':[]}
  def forward(self,x):
    x=self.relu(self.fc1(x))
    x=self.relu(self.fc2(x))
    x=self.fc3(x)
    return x



# Initialize the Model, Loss Function, and Optimizer
def train_model(Aishwariya_brain=_brain,x_train,y_train,criterion,optimizer,epochs=2000):
  for epoch in range(epochs):
    optimizer.zero_grad()
    loss=criterion(Aadithyan_brain(x_train),y_train)
    loss.backward()
    optimizer.step()
    Aadithyan_brain.history['loss'].append(loss.item())
    if epoch%200==0:
      print(f'epoch:[{epoch}/{epochs}], loss:{loss.item():.6f}')



```
## Dataset Information

<img width="549" height="685" alt="image" src="https://github.com/user-attachments/assets/8c0ca5b3-772b-4f23-89f3-b6a362e578c4" />
## OUTPUT

### Training Loss Vs Iteration Plot

<img width="866" height="566" alt="Screenshot 2025-09-16 112358" src="https://github.com/user-attachments/assets/52a1e0bd-ca84-45f5-9108-feb11204164c" />


## RESULT
The program to develop a neural network regression model for the given dataset has been executed successively
