import cupy as cp
import math
import time

# Get the default device ID (usually 0)
device_id = cp.cuda.runtime.getDevice()

# Get device properties
props = cp.cuda.runtime.getDeviceProperties(device_id)

# Print device name
device_name = props['name'].decode('utf-8')
print(f"Device Name: {device_name}")

# Print compute capability
cc_major = props['major']
cc_minor = props['minor']
print(f"Compute Capability: {cc_major}.{cc_minor}")

# Print total memory
total_mem = props['totalGlobalMem'] / (1024 ** 3)  # Convert bytes to GB
print(f"Total Memory: {total_mem:.2f} GB")

def softmax(x):
    matrix_exp = cp.exp(x - cp.max(x, axis=0, keepdims=True))

    # Step 2: Compute the sum of exponentials for each column
    column_sums = cp.sum(matrix_exp, axis=0, keepdims=True)

    # Step 3: Divide each element by the sum of the exponentials of its column
    return matrix_exp / column_sums


def forward_prop(input, weights):
                   
    x1 = cp.dot(weights[0], input)
    A1 = cp.maximum(0, x1) #apply relu

    #append 1s to beginning to multiply with biases
    temp = cp.ones((1, x1.shape[1]))
    A1_with_bias = cp.vstack((temp, A1))

    x2 = cp.dot(weights[1], A1_with_bias)
    A2 = cp.maximum(0, x2) #apply relu

    #append 1s to beginning to multiply with biases
    temp = cp.ones((1, x2.shape[1]))
    A2_with_bias = cp.vstack((temp, A2))

    x3 = cp.dot(weights[2], A2_with_bias)

    return x1, A1_with_bias, x2, A2_with_bias, softmax(x3)

def categorical_crossentropy(output, true_labels):
    epsilon = 1e-12
    Y_pred = cp.clip(output, epsilon, 1. - epsilon)

    # Step 2: Compute the log of predicted probabilities
    log_Y_pred = cp.log(Y_pred)

    # Step 3: Compute element-wise multiplication between Y_true and log_Y_pred
    return -cp.sum(true_labels * log_Y_pred, axis=0)

def caclulate_accuracy(output, true_labels):
    predicted_classes = cp.argmax(output, axis=0)

    # Step 2: Convert one-hot encoded true labels to class indices (index of 1 for each column)
    true_classes = cp.argmax(true_labels, axis=0)

    # Step 3: Compare predicted classes with true classes
    correct_predictions = cp.sum(predicted_classes == true_classes)

    # Step 4: Compute accuracy
    return correct_predictions / true_labels.shape[1]

def categorical_crossentropy_derivative(Y_true, Y_pred):
    return Y_pred - Y_true  # Gradient of softmax and cross-entropy combined

def relu_derivative(Z):
    return Z > 0


def backprop(input, y_train, outputs, weights, alpha, m, v, beta_1, beta_2, epsilon, t):
    
    num_samples = outputs[4].shape[1]
    
    d_output = categorical_crossentropy_derivative(y_train, outputs[4])  # Shape (10, batch_size)
    dW2 = cp.dot(d_output, outputs[3].T) / num_samples

    dA1 = cp.dot(weights[2][:, 1:].T, d_output)  # Backprop into hidden layer, ignoring bias term
    dZ1 = dA1 * relu_derivative(outputs[2])  # Apply ReLU derivative

    dW1 = cp.dot(dZ1, outputs[1].T) / num_samples  # (hidden_size, n_features + 1)

    dA0 = cp.dot(weights[1][:, 1:].T, dZ1)  # Backprop into hidden layer 1, ignoring bias term
    dZ0 = dA0 * relu_derivative(outputs[0])  # Apply ReLU derivative

    dW0 = cp.dot(dZ0, input.T) / num_samples  # (hidden_size1, n_features + 1)

    # ADAM updates
    
    for i, dW in enumerate([dW0, dW1, dW2]):
        # Update biased first moment estimate
        m[i] = beta_1 * m[i] + (1 - beta_1) * dW
        # Update biased second raw moment estimate
        v[i] = beta_2 * v[i] + (1 - beta_2) * (dW ** 2)

        # Compute bias-corrected first moment estimate
        m_hat = m[i] / (1 - beta_1 ** t)
        # Compute bias-corrected second raw moment estimate
        v_hat = v[i] / (1 - beta_2 ** t)

        # Update weights
        weights[i] -= alpha * m_hat / (cp.sqrt(v_hat) + epsilon)

def execute_model(X, Y, weights, alpha, m, v, beta_1, beta_2, epsilon, t):
    
    start = time.time()
    #test forward prop
    output = forward_prop(X, weights)
    #print(output[4].shape)

    backprop(X, Y, output, weights, alpha, m, v, beta_1, beta_2, epsilon, t)
    #print(weights[0].shape, weights[1].shape, weights[2].shape)

    end = time.time()

    #calculate loss
    loss = categorical_crossentropy(output[4], Y)
    #print("Loss: ", cp.mean(loss))

    #calculate accuracy
    accuracy = caclulate_accuracy(output[4], Y)
    #print("Accuracy: ", cp.mean(accuracy))

    return cp.hstack((end - start, cp.mean(loss), cp.mean(accuracy)))