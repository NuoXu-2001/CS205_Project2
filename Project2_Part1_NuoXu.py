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
    num_features = X.shape[1]
    selected = set() # Currently selected features

    best_global_acc = 0.0
    best_global_set = set()
    prev_best_acc = 0.0 # Track previous level's accuracy for warning detection

    print(f"This dataset has {num_features} features (not including the class attribute), with {X.shape[0]} instances.")
    print()

    # Show baseline: accuracy with all features
    all_features_acc = loo_accuracy(X, y, set(range(num_features)))
    print(f'Running nearest neighbor with all {num_features} features, using "leaving-one-out" evaluation, I get an accuracy of {all_features_acc:.1%}')
    print()
    print("Beginning search.")
    print()

    for level in range(num_features):
        best_acc = -1 # Best accuracy for this level
        best_feature = -1 # Best feature to add at this level

        # Add each remaining feature and evaluate performance
        for feature in range(num_features):
            if feature not in selected: # Only consider unselected features
                test_set = selected | {feature} # Create test set by adding this feature to current selection
                accuracy = loo_accuracy(X, y, test_set) # Evaluate accuracy

                # Show this trial
                feature_list = sorted([f+1 for f in test_set]) # Convert to 1-based indexing
                print(f"Using feature(s) {{{','.join(map(str, feature_list))}}} accuracy is {accuracy:.1%}")

                # Track best feature for this level
                if accuracy > best_acc:
                    best_acc = accuracy
                    best_feature = feature

        # Add the best feature found at this level
        if best_feature != -1:
            selected.add(best_feature)
            feature_list = sorted([f+1 for f in selected])

            # Check for accuracy decrease
            if level > 0 and best_acc < prev_best_acc:
                print("(Warning, Accuracy has decreased! Continuing search in case of local maxima)")

            # Show the choice made at this level
            print(f"Feature set {{{','.join(map(str, feature_list))}}} was best, accuracy is {best_acc:.1%}")
            print()

            # Update global best if current combination is better
            if best_acc > best_global_acc:
                best_global_acc = best_acc
                best_global_set = selected.copy()

            prev_best_acc = best_acc

        else:
            break # No improvement, stop search

    # Show the final result
    best_feature_list = sorted([f+1 for f in best_global_set])
    print(f"Finished search!! The best feature subset is {{{','.join(map(str, best_feature_list))}}}, which has an accuracy of {best_global_acc:.1%}")
    return best_global_set, best_global_acc

def backward_elimination(X, y):
    num_features = X.shape[1]
    remaining = set(range(num_features)) # Start with all features

    print(f"This dataset has {num_features} features (not including the class attribute), with {X.shape[0]} instances.")
    print()

    # Show baseline: accuracy with all features
    current_acc = loo_accuracy(X, y, remaining)
    print(f'Running nearest neighbor with all {num_features} features, using "leaving-one-out" evaluation, I get an accuracy of {current_acc:.1%}')
    print()
    print("Beginning search.")
    print()

    # Global tracking
    best_global_acc = current_acc
    best_global_set = remaining.copy()
    prev_best_acc = current_acc # Track previous level's accuracy

    while len(remaining) > 1: # Continue until only one feature remains
        best_acc = -1 # Best accuracy for this level
        worst_feature = -1 # Feature whose removal gives best improvement

        # Remove each remaining feature and evaluate performance
        for feature in list(remaining): 
            test_set = remaining - {feature} # Create test set by removing this feature from current selection

            # Evaluate accuracy without this feature
            accuracy = loo_accuracy(X, y, test_set)

            # Show this trial
            feature_list = sorted([f+1 for f in test_set])  # Convert to 1-based indexing
            print(f"Using feature(s) {{{','.join(map(str, feature_list))}}} accuracy is {accuracy:.1%}")
            
            # Track best feature to remove at this level
            if accuracy > best_acc:
                best_acc = accuracy
                worst_feature = feature

        # Remove the worst feature found at this level
        if worst_feature != -1:
            remaining.remove(worst_feature)
            feature_list = sorted([f+1 for f in remaining])

            # Check for accuracy decrease
            if best_acc < prev_best_acc:
                print("(Warning, Accuracy has decreased! Continuing search in case of local maxima)")

            # Show the choice made at this level
            print(f"Feature set {{{','.join(map(str, feature_list))}}} was best, accuracy is {best_acc:.1%}")
            print()

            # Update global best if current combination is better
            if best_acc > best_global_acc:
                best_global_acc = best_acc
                best_global_set = remaining.copy()

            prev_best_acc = best_acc
        else:
            break # No improvement, stop search
    
    # Show the final result
    best_feature_list = sorted([f+1 for f in best_global_set])
    print(f"Finished search!! The best feature subset is {{{','.join(map(str, best_feature_list))}}}, which has an accuracy of {best_global_acc:.1%}")
    return best_global_set, best_global_acc

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