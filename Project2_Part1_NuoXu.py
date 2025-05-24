import numpy as np

def load_data(filename):
    data = np.loadtxt(filename)
    return data[:, 1:], data[:, 0].astype(int) #features, classes

def nearest_neighbor_classify(train_X, train_y, test_point):
    distances = np.linalg.norm(train_X - test_point, axis = 1) # Calculate Euclidean distance from test point to all training points
    nearest_idx = np.argmin(distances) # Find index of nearest neighbor
    return train_y[nearest_idx] # Return class label of nearest neighbor

def loo_accuracy(X, y, feature_indices):
    if not feature_indices: # No features selected = 0% accuracy
        return 0.0
    
    X_subset = X[:, list(feature_indices)] # Extract only the selected features
    correct = 0
    
    # Leave-one-out cross validation
    for i in range(len(X_subset)):
        # Create training set by removing i-th sample
        train_X = np.delete(X_subset, i, axis=0)
        train_y = np.delete(y, i)

        # Test on i-th sample
        test_point = X_subset[i]

        # Classify using nearest neighbour
        predicted = nearest_neighbor_classify(train_X, train_y, test_point)

        # Count correct predictions
        if predicted == y[i]:
            correct += 1

    return correct / len(X_subset)

def forward_selection(X, y):

def backward_elimination(X, y):

def main():
    print("Welcome to My Feature Selection Algorithm.")
    filename = input("Type in the name of the file to test: ")
    X, y = load_data(filename) # Load dataset

    # Show algorithm choices to user
    print("\nType the number of the algorithm you want to run.")
    print("1) Forward Selection")
    print("2) Backward Elimination")
    choice = input()
    print()

    # Execute selected algorithm
    if choice == '1':
        features, accuracy = forward_selection(X, y)
    elif choice == '2':
        features, accuracy = backward_elimination(X, y)
    else:
        print("Invalid choice.")
        return
    
if __name__ == "__main__":
    main()