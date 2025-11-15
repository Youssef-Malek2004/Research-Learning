from GradientDescentOptimizer import GradientDescentOptimizer
from NeuralLayer import NeuralLayer, ReLULayer
from Micrograd import Value
import pandas as pd

def main():
    lr = 0.001

    linear1 = NeuralLayer(2,4)
    linear2 = NeuralLayer(4, 2)
    linear3 = NeuralLayer(2, 1)

    params = []
    print(linear1.parameters())
    print(linear2.parameters())
    print(linear3.parameters())
    params += linear1.parameters()
    params += linear2.parameters()
    params += linear3.parameters()

    relu = ReLULayer()

    optimizer = GradientDescentOptimizer(params, lr)

    epochs = 100

    df = pd.read_csv('synthetic_dataset.csv')

    total_loss = 0.0

    for epoch in range(epochs):
        printed = False
        for i in range(df.shape[0]):
            row = df.iloc[i]

            inputs = [Value(data=row['x1']), Value(data=row['x2'])]

            pred = linear3(relu(linear2(relu(linear1(inputs)))))
            true = Value(data=row['y'])

            loss = (true - pred[0]) ** 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.data

            if epoch % 10 == 0 and printed == False:
                print(f"Total Loss @ Epoch {epoch}: {loss.data}")
                printed = True

    # print(neuron.parameters())
    # print(neuron.weights)
    # print(neuron.bias)

if __name__ == "__main__":
    main()