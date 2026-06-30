import pandas as pd
import numpy as np

# Load the csv file from the folder
data_file = pd.read_csv('Stress Indicators Dataset for Mental Health Classification.csv')

# The data is really uneven so I am oversampling the small groups
# so the model can actually learn what high and low stress looks like
group_high = data_file[data_file['stress_type'] == 0]
group_mid  = data_file[data_file['stress_type'] == 1]
group_low  = data_file[data_file['stress_type'] == 2]

# Making them all the same size as the biggest group
target_size = len(group_mid)
high_resampled = group_high.sample(target_size, replace=True, random_state=42)
low_resampled  = group_low.sample(target_size, replace=True, random_state=42)
balanced_data  = pd.concat([group_mid, high_resampled, low_resampled])

# Split into X (features) and y (target)
features = [col for col in data_file.columns if col != 'stress_type']
X = balanced_data[features].values

# Flip the target so 1.0 is high stress and 0.0 is low stress
y = 1.0 - (balanced_data['stress_type'].values.reshape(-1, 1).astype(float) / 2.0)

# Calculate math values for scaling
dataset_medians = np.median(X, axis=0)
dataset_std_devs = np.std(X, axis=0) + 1e-8

def scale_numbers(input_data):
    # This scales everything and stops crazy extreme values from breaking the model
    scaled = (input_data - dataset_medians) / dataset_std_devs
    return np.clip(scaled, -3.0, 3.0)

# Prepare the data for training
X_scaled = scale_numbers(X)
np.random.seed(42)
shuffled_idx = np.random.permutation(len(X_scaled))
X_train, y_train = X_scaled[shuffled_idx], y[shuffled_idx]

# Simple Neural Network class
class StudentStressModel:
    def __init__(self, input_size, hidden_nodes=64, l_rate=0.05):
        # Initialize weights with some random numbers
        self.w1 = np.random.randn(input_size, hidden_nodes) * np.sqrt(2./input_size)
        self.b1 = np.zeros((1, hidden_nodes))
        self.w2 = np.random.randn(hidden_nodes, 1) * np.sqrt(2./hidden_nodes)
        self.b2 = np.zeros((1, 1))
        self.learning_rate = l_rate

    def sigmoid_func(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -20, 20)))

    def forward(self, x_input):
        # Calculate the path through the layers
        self.z1 = x_input @ self.w1 + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = self.a1 @ self.w2 + self.b2
        self.output = self.sigmoid_func(self.z2)
        return self.output

    def train(self, x_data, y_data, epochs=2000):
        m = x_data.shape[0]
        for i in range(epochs):
            # Forward pass
            predictions = self.forward(x_data)
           
            # Backpropagation (finding the errors)
            error_val = (predictions - y_data)
            dz2 = error_val * (predictions * (1 - predictions))
            dw2 = (self.a1.T @ dz2) / m
            db2 = np.mean(dz2, axis=0, keepdims=True)
           
            dz1 = (dz2 @ self.w2.T) * (1 - self.a1**2)
            dw1 = (x_data.T @ dz1) / m
            db1 = np.mean(dz1, axis=0, keepdims=True)

            # Gradient clipping to keep updates stable
            for grad in [dw1, db1, dw2, db2]:
                np.clip(grad, -0.5, 0.5, out=grad)

            # Update the weights
            self.w1 -= self.learning_rate * dw1
            self.b1 -= self.learning_rate * db1
            self.w2 -= self.learning_rate * dw2
            self.b2 -= self.learning_rate * db2

# Create and train the model
my_model = StudentStressModel(input_size=len(features))
print("Training model... please wait...")
my_model.train(X_train, y_train)
print("Training finished!\n")

def take_survey():
    print("--- Personal Stress Level Survey ---")
    responses = []
   
    for f_name in features:
        if 'gender' in f_name.lower():
            text = "Gender (Input 1 for Male, 0 for Female): "
        else:
            text = f"{f_name.replace('_', ' ').title()}: "
           
        try:
            val = float(input(text))
           
            # Handle extreme values or typos by capping them
            if 'age' in f_name.lower():
                val = np.clip(val, 10, 100)
            elif 'gender' in f_name.lower():
                val = np.clip(val, 0, 1)
            else:
                val = np.clip(val, 1, 5) # Likert scale
            responses.append(val)
        except:
            # If user types something weird, use the middle value
            responses.append(dataset_medians[features.index(f_name)])

    # Scale the input and get the prediction
    test_input = scale_numbers(np.array(responses).reshape(1, -1))
    prob_result = my_model.forward(test_input)[0][0]
   
    # Scale result to be 0 to 100
    final_score = prob_result * 100

    print("\n" + "="*30)
    print(f"STRESS SCORE: {final_score:.2f} / 100")
   
    if final_score >= 80:
        status = "CRITICAL STRESS"
    elif final_score >= 60:
        status = "HIGH STRESS"
    elif final_score >= 40:
        status = "MODERATE STRESS"
    else:
        status = "LOW STRESS"
       
    print(f"RESULT: {status}")
    print("Scale: 0 (Low) to 100 (High)")
    print("="*30)

# Run the survey
take_survey()