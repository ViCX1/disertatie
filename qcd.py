import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pickle
from sklearn.model_selection import KFold




def preprocess_data(df, N=10):
    # Copy the dataframe to avoid modifying the original one
    df_processed = df.copy()

    df_processed = df_processed.fillna(0)

    # Process 'gps_x' and 'gps_y' columns: remove dots
    '''if 'gps_x' in df_processed.columns:
        df_processed['gps_x'] = df_processed['gps_x'].astype(str).str.replace('.', '').astype(float)
    else:
        print("Column 'gps_x' not found. Skipping preprocessing for this column.")

    if 'gps_y' in df_processed.columns:
        df_processed['gps_y'] = df_processed['gps_y'].astype(str).str.replace('.', '').astype(float)
    else:
        print("Column 'gps_y' not found. Skipping preprocessing for this column.")
'''
    # Process 'wifi_ap' column: remove dots
    if 'wifi_ap' in df_processed.columns:
        df_processed['wifi_ap'] = df_processed['wifi_ap'].astype(str).str.replace('.', '').astype(float)
    else:
        print("Column 'wifi_ap' not found. Skipping preprocessing for this column.")

    # Process 'bt_mac' column: convert MAC addresses to decimal numbers
    if 'bt_mac' in df_processed.columns:
        df_processed['bt_mac'] = df_processed['bt_mac'].astype(str).apply(lambda x: int(x.replace('-', ''), 16))
    else:
        print("Column 'bt_mac' not found. Skipping preprocessing for this column.")

    # Process 'date' column: remove dots
    if 'date' in df_processed.columns:
        df_processed['date'] = df_processed['date'].astype(str).str.replace('.', '').astype(int)
    else:
        print("Column 'date' not found. Skipping preprocessing for this column.")

    
    # Select only the numeric columns
    if 'label' in df.columns:
        df = df.drop(columns=['label'])

    df_processed = df_processed.select_dtypes(include=[np.number])


    scaler = StandardScaler()
    df_processed = pd.DataFrame(scaler.fit_transform(df_processed), columns=df_processed.columns)


    return df_processed

 

def plot_model_performance(y_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(y_pred)), y_pred, 'r-')
    plt.title('Model Predictions')
    plt.xlabel('Sample Index')
    plt.ylabel('Prediction')
    plt.show()



directory = '/Users/user/Desktop/qcd/training/'

df_train = pd.concat([pd.read_csv(directory + filename, delimiter=';') for filename in os.listdir(directory) if filename.endswith('.csv')])

# Separate labels from the rest of the data
X_train = preprocess_data(df_train.drop(columns=['label']))
y_train = df_train['label']

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Check if best parameters are already saved
if os.path.isfile('best_params.pickle'):
    # Load the best parameters from a file
    with open('best_params.pickle', 'rb') as f:
        best_params = pickle.load(f)

    # Create a RandomForestClassifier model with the best parameters
    model = RandomForestClassifier(n_estimators=best_params["n_estimators"], 
                                   max_features=best_params["max_features"],
                                   max_depth=best_params["max_depth"],
                                   min_samples_split=best_params["min_samples_split"],
                                   min_samples_leaf=best_params["min_samples_leaf"],
                                   bootstrap=best_params["bootstrap"])
    # Cross-validation for monitoring accuracy and precision during training
    kf = KFold(n_splits=5)
    accuracy = []
    precision = []
    
    for train_index, val_index in kf.split(X_train):
        X_train_kf, X_val_kf = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_kf, y_val_kf = y_train.iloc[train_index], y_train.iloc[val_index]
        
        model.fit(X_train_kf, y_train_kf)
        predictions = model.predict(X_val_kf)
        
        accuracy.append(accuracy_score(y_val_kf, predictions))
        precision.append(precision_score(y_val_kf, predictions))
        
    # Plot accuracy and precision
    plt.figure(figsize=(10,6))
    plt.plot(range(1, len(accuracy)+1), accuracy, label='Accuracy')
    plt.plot(range(1, len(precision)+1), precision, label='Precision')
    plt.title('Accuracy and Precision over Cross-Validation Folds')
    plt.xlabel('Folds')
    plt.ylabel('Score')
    plt.legend()
    plt.show()


else:
    # Best parameters are not saved, need to run the RandomizedSearchCV

    # Use the random grid to search for best hyperparameters
    rf = RandomForestClassifier()

    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    # Fit the random search model
    rf_random.fit(X_train, y_train)

    # The best parameters can be obtained with:
    best_params = rf_random.best_params_

    # Save the best parameters to a file
    with open('best_params.pickle', 'wb') as f:
        pickle.dump(best_params, f)

    # print the best parameters
    print("Best parameters found: ", best_params)

    # Create a RandomForestClassifier model with the best parameters
    model = RandomForestClassifier(n_estimators=best_params["n_estimators"], 
                                   max_features=best_params["max_features"],
                                   max_depth=best_params["max_depth"],
                                   min_samples_split=best_params["min_samples_split"],
                                   min_samples_leaf=best_params["min_samples_leaf"],
                                   bootstrap=best_params["bootstrap"])
    model.fit(X_train, y_train)

directory2 = '/Users/user/Desktop/qcd/testing/i/'

# Testing phase
if os.path.isfile('testing2.csv'):
    #df_test = pd.read_csv('testing2.csv', delimiter=';')

    

    df_test = pd.concat([pd.read_csv(directory2 + filename, delimiter=';') for filename in os.listdir(directory2) if filename.endswith('.csv')])

    X_test = preprocess_data(df_test.drop(columns=['label']))
    y_test = df_test['label']

    # Make predictions
    y_pred = model.predict(X_test)

    print(np.unique(y_pred))


    # Calculate the F1 score
    f1 = f1_score(y_test, y_pred)

    # Print the F1 score
    print(f'F1 Score: {f1*100:.2f}%')



# Calculate the confusion matrix
unique_preds = np.unique(y_pred)
if len(unique_preds) == 2:
    cm = confusion_matrix(y_test, y_pred)
    # Create a DataFrame from the confusion matrix so that it's easier to plot
    cm_df = pd.DataFrame(cm, columns=['Predicted Negative', 'Predicted Positive'], 
                         index=['Actual Negative', 'Actual Positive'])

    # Plot the confusion matrix
    sns.heatmap(cm_df, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.show()
else:
    print(f"Only one class ({unique_preds[0]}) predicted, unable to create confusion matrix")
    cm_df = None  # cm_df needs to be defined in both cases
    
# Calculate the percentage of testing samples predicted as user data
user_percentage = (y_pred.sum() / len(y_pred)) * 100
print(f"Percentage of testing samples predicted as user data: {user_percentage:.2f}%")

# Calculate and print accuracy and precision
if 'label' in df_test.columns:
    y_test = df_test['label']
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    print(f'Accuracy: {accuracy*100:.2f}%')
    print(f'Precision: {precision*100:.2f}%')

    # Compute ROC curve and ROC area for each class
    y_pred_proba = model.predict_proba(X_test)
    if y_pred_proba.shape[1] == 1:
        print("Only one class predicted, unable to compute ROC curve")
    else:
        y_pred_proba = y_pred_proba[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        # Plot of a ROC curve for a specific class
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()


    nperc = 90

    if user_percentage > nperc:
        print("The testing data is predicted to be from the same user.")
    else:
        print("The testing data is predicted to be from a different user.")
        
    # Plot model performance
    plot_model_performance(y_pred)
else:
    print("No testing file found. Skipping testing phase.")