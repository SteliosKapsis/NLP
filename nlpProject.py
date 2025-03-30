import pandas as pd

# Load the CSV file into a Pandas DataFrame
file_path = 'IMDB-Dataset.csv'

data = pd.read_csv(file_path)

# Display the first few rows of the dataframe to understand its structure
print(data.head())

# Add the 'label' column based on the 'sentiment' column
data['label'] = data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0 if x == 'negative' else None)

# Display the first few rows of the dataframe to check the new column
print(data.head())

# Keep only the 'review' and 'label' columns
data = data[['review', 'label']]

# Display the first few rows of the dataframe to check
print(data.head())

import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Ensure stopwords and wordnet are downloaded
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Text cleaning function
def preprocess_text(text):
    text = text.lower()  # Lowercasing
    text = re.sub(r'<br\s*/?>', ' ', text)  # Remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters and numbers
    tokens = text.split()  # Tokenization
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # Lemmatization and stopword removal
    return ' '.join(tokens)

# Apply preprocessing
data['cleaned_review'] = data['review'].apply(preprocess_text)

# Keep only the necessary columns
data_cleaned = data[['cleaned_review', 'label']]

# Display the first few rows of the cleaned data
data_cleaned.head()

from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))  # Using bigrams and limiting features for efficiency
X = vectorizer.fit_transform(data_cleaned['cleaned_review'])
y = data_cleaned['label']

from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
import numpy as np

# Initialize Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Prepare the data
X = vectorizer.fit_transform(data_cleaned['cleaned_review'])
y = data_cleaned['label']

# Create arrays for storing results
X_train_folds, X_test_folds, y_train_folds, y_test_folds = [], [], [], []

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Apply SMOTE only on the training set if needed
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Store the resampled data
    X_train_folds.append(X_train_resampled)
    y_train_folds.append(y_train_resampled)
    X_test_folds.append(X_test)
    y_test_folds.append(y_test)

# Confirm fold sizes
for i, (X_train_fold, y_train_fold) in enumerate(zip(X_train_folds, y_train_folds)):
    print(f"Fold {i+1} - Training set size: {X_train_fold.shape[0]}, Test set size: {X_test_folds[i].shape[0]}")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=1000)

# To store results for averaging
accuracy_scores_lr, precision_scores_lr, recall_scores_lr, f1_scores_lr = [], [], [], []
tprs_lr = []
aucs_lr = []
mean_fpr_lr = np.linspace(0, 1, 100)

# Loop through each fold of the cross-validation
for i, (X_train_fold, y_train_fold, X_test_fold, y_test_fold) in enumerate(zip(X_train_folds, y_train_folds, X_test_folds, y_test_folds)):
    model.fit(X_train_fold, y_train_fold)
    y_pred = model.predict(X_test_fold)
    
    # Generate classification report as a dictionary
    report = classification_report(y_test_fold, y_pred, output_dict=True)
    
    # Calculate accuracy for Logistic Regression
    accuracy = accuracy_score(y_test_fold, y_pred)
    accuracy_scores_lr.append(accuracy)
    
    # Store metrics for the positive class (label 1)
    precision_scores_lr.append(report['1']['precision'])
    recall_scores_lr.append(report['1']['recall'])
    f1_scores_lr.append(report['1']['f1-score'])
    
    # Display the metrics for this fold
    print(f"\nFold {i+1} Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {report['1']['precision']:.4f}")
    print(f"Recall: {report['1']['recall']:.4f}")
    print(f"F1-Score: {report['1']['f1-score']:.4f}")
    
    # Compute ROC curve and AUC for Logistic Regression
    y_prob = model.predict_proba(X_test_fold)[:, 1]
    fpr, tpr, _ = roc_curve(y_test_fold, y_prob)
    tprs_lr.append(np.interp(mean_fpr_lr, fpr, tpr))
    tprs_lr[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs_lr.append(roc_auc)

    print(f"AUC: {roc_auc:.4f}")

# Calculate and print average scores across all folds for Logistic Regression
print(f"\nAverage Accuracy: {np.mean(accuracy_scores_lr):.4f}")
print(f"Average Precision: {np.mean(precision_scores_lr):.4f}")
print(f"Average Recall: {np.mean(recall_scores_lr):.4f}")
print(f"Average F1-Score: {np.mean(f1_scores_lr):.4f}")
print(f"Average AUC: {np.mean(aucs_lr):.4f}")

# Plot the ROC curve for Logistic Regression
plt.figure(figsize=(10, 8))
mean_tpr_lr = np.mean(tprs_lr, axis=0)
mean_tpr_lr[-1] = 1.0
mean_auc_lr = auc(mean_fpr_lr, mean_tpr_lr)
plt.plot(mean_fpr_lr, mean_tpr_lr, color='b', label=f'Mean ROC (AUC = {mean_auc_lr:.2f})')
plt.fill_between(mean_fpr_lr, np.percentile(tprs_lr, 25, axis=0), np.percentile(tprs_lr, 75, axis=0), color='blue', alpha=0.2, label='±1 std. dev.')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
plt.title('ROC Curve - Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid()
plt.show()

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt

# Initialize the Random Forest model
model = RandomForestClassifier(random_state=42)

# To store results for averaging
accuracy_scores_rf, precision_scores_rf, recall_scores_rf, f1_scores_rf = [], [], [], []
tprs_rf = []
aucs_rf = []
mean_fpr_rf = np.linspace(0, 1, 100)

# Loop through each fold of the cross-validation
for i, (X_train_fold, y_train_fold, X_test_fold, y_test_fold) in enumerate(zip(X_train_folds, y_train_folds, X_test_folds, y_test_folds)):
    model.fit(X_train_fold, y_train_fold)
    y_pred = model.predict(X_test_fold)
    
    # Generate classification report as a dictionary
    report = classification_report(y_test_fold, y_pred, output_dict=True)
    
    # Calculate accuracy for Random Forest
    accuracy = accuracy_score(y_test_fold, y_pred)
    accuracy_scores_rf.append(accuracy)
    
    # Store metrics for the positive class (label 1)
    precision_scores_rf.append(report['1']['precision'])
    recall_scores_rf.append(report['1']['recall'])
    f1_scores_rf.append(report['1']['f1-score'])
    
    # Display the metrics for this fold
    print(f"\nFold {i+1} Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {report['1']['precision']:.4f}")
    print(f"Recall: {report['1']['recall']:.4f}")
    print(f"F1-Score: {report['1']['f1-score']:.4f}")
    
    # Compute ROC curve and AUC for Random Forest
    y_prob = model.predict_proba(X_test_fold)[:, 1]
    fpr, tpr, _ = roc_curve(y_test_fold, y_prob)
    tprs_rf.append(np.interp(mean_fpr_rf, fpr, tpr))
    tprs_rf[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs_rf.append(roc_auc)

    print(f"AUC: {roc_auc:.4f}")

# Calculate and print average scores across all folds for Random Forest
print(f"\nAverage Accuracy: {np.mean(accuracy_scores_rf):.4f}")
print(f"Average Precision: {np.mean(precision_scores_rf):.4f}")
print(f"Average Recall: {np.mean(recall_scores_rf):.4f}")
print(f"Average F1-Score: {np.mean(f1_scores_rf):.4f}")
print(f"Average AUC: {np.mean(aucs_rf):.4f}")

# Plot the ROC curve for Random Forest
plt.figure(figsize=(10, 8))
mean_tpr_rf = np.mean(tprs_rf, axis=0)
mean_tpr_rf[-1] = 1.0
mean_auc_rf = auc(mean_fpr_rf, mean_tpr_rf)
plt.plot(mean_fpr_rf, mean_tpr_rf, color='b', label=f'Mean ROC (AUC = {mean_auc_rf:.2f})')
plt.fill_between(mean_fpr_rf, np.percentile(tprs_rf, 25, axis=0), np.percentile(tprs_rf, 75, axis=0), color='blue', alpha=0.2, label='±1 std. dev.')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
plt.title('ROC Curve - Random Forest')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# Create a summary dataframe for the models' performance
results = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'],
    'Logistic Regression': [
        np.mean(accuracy_scores_lr), np.mean(precision_scores_lr), np.mean(recall_scores_lr),
        np.mean(f1_scores_lr), np.mean(aucs_lr)
    ],
    'Random Forest': [
        np.mean(accuracy_scores_rf), np.mean(precision_scores_rf), np.mean(recall_scores_rf),
        np.mean(f1_scores_rf), np.mean(aucs_rf)
    ]
}

# Create a DataFrame to display the results
comparison_df = pd.DataFrame(results)
print(comparison_df)

import matplotlib.pyplot as plt

# Prepare data for plotting
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
log_reg_scores = [np.mean(accuracy_scores_lr), np.mean(precision_scores_lr), np.mean(recall_scores_lr),
                  np.mean(f1_scores_lr), np.mean(aucs_lr)]
rf_scores = [np.mean(accuracy_scores_rf), np.mean(precision_scores_rf), np.mean(recall_scores_rf),
             np.mean(f1_scores_rf), np.mean(aucs_rf)]

# Plot bar chart for comparison
x = range(len(metrics))
width = 0.35  # width of bars

plt.figure(figsize=(10, 6))

# Bar chart for Logistic Regression and Random Forest
plt.bar(x, log_reg_scores, width, label='Logistic Regression', color='b', alpha=0.7)
plt.bar([p + width for p in x], rf_scores, width, label='Random Forest', color='g', alpha=0.7)

# Add labels, title, and legend
plt.xlabel('Metrics')
plt.ylabel('Scores')
plt.title('Model Comparison: Logistic Regression vs Random Forest')
plt.xticks([p + width / 2 for p in x], metrics)
plt.legend()

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc

# Calculate mean TPR and AUC for Logistic Regression
mean_tpr_lr = np.mean(tprs_lr, axis=0)
mean_tpr_lr[-1] = 1.0  # Ensure the ROC curve ends at (1, 1)
mean_auc_lr = auc(mean_fpr_lr, mean_tpr_lr)

# Calculate mean TPR and AUC for Random Forest
mean_tpr_rf = np.mean(tprs_rf, axis=0)
mean_tpr_rf[-1] = 1.0  # Ensure the ROC curve ends at (1, 1)
mean_auc_rf = auc(mean_fpr_rf, mean_tpr_rf)

# Plot the ROC curves for both models
plt.figure(figsize=(10, 8))

# Logistic Regression ROC curve
plt.plot(
    mean_fpr_lr, mean_tpr_lr, color='blue',
    label=f'Logistic Regression (AUC = {mean_auc_lr:.2f})'
)
plt.fill_between(
    mean_fpr_lr,
    np.percentile(tprs_lr, 25, axis=0),
    np.percentile(tprs_lr, 75, axis=0),
    color='blue', alpha=0.2, label='Logistic Regression ±1 std. dev.'
)

# Random Forest ROC curve
plt.plot(
    mean_fpr_rf, mean_tpr_rf, color='green',
    label=f'Random Forest (AUC = {mean_auc_rf:.2f})'
)
plt.fill_between(
    mean_fpr_rf,
    np.percentile(tprs_rf, 25, axis=0),
    np.percentile(tprs_rf, 75, axis=0),
    color='green', alpha=0.2, label='Random Forest ±1 std. dev.'
)

# Plot the diagonal (random guess line)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')

# Add title, labels, and legend
plt.title('ROC Curve Comparison - Logistic Regression vs Random Forest')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid()
plt.tight_layout()

# Display the plot
plt.show()

#########################################################################
#########################################################################
######################### DEEP LEARNING MODELS ##########################
#########################################################################
#########################################################################

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

# Custom Dataset class to handle the data
class SentimentDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.toarray(), dtype=torch.float32)  # Converting sparse matrix to dense
        self.y = torch.tensor(y.values, dtype=torch.long)  # Labels as long type for classification
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Creating DataLoaders for training and testing
def create_data_loaders(X_train, y_train, X_test, y_test, batch_size=64):
    train_dataset = SentimentDataset(X_train, y_train)
    test_dataset = SentimentDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# CNN model
class CNNModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CNNModel, self).__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=5)  # 1 channel input
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc = nn.Linear(128 * (input_dim - 4) // 2, output_dim)  # Adjusted to input size after pooling
    
    def forward(self, x):
        x = x.unsqueeze(1)  # Adding a channel dimension: shape becomes (batch_size, 1, seq_length)
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, features)
        x = self.fc(x)
        return x

# Define the MLP model class
class MLPModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPModel, self).__init__()
        
        # Define the architecture of the MLP
        self.fc1 = nn.Linear(input_dim, 128)  # First fully connected layer
        self.fc2 = nn.Linear(128, 64)         # Second fully connected layer
        self.fc3 = nn.Linear(64, output_dim)  # Output layer
        
        self.relu = nn.ReLU()  # Activation function
        self.softmax = nn.Softmax(dim=1)  # Softmax for multi-class classification
        
    def forward(self, x):
        # Forward pass through the layers with activations
        x = self.relu(self.fc1(x))  # Apply ReLU activation after first layer
        x = self.relu(self.fc2(x))  # Apply ReLU activation after second layer
        x = self.fc3(x)             # Output layer without activation (for CrossEntropyLoss)
        return x

def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Print epoch statistics
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

# Evaluation function to calculate accuracy, precision, recall, F1-score, and AUC
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # For ROC and AUC
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.numpy())
    
    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    auc_score = roc_auc_score(all_labels, all_probs)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    return accuracy, precision, recall, f1, auc_score, fpr, tpr

# CNN Model Training and Evaluation
all_accuracies_cnn, all_precisions_cnn, all_recalls_cnn, all_f1_scores_cnn, all_aucs_cnn = [], [], [], [], []
mean_fpr_cnn = np.linspace(0, 1, 100)  # 100 evenly spaced points for FPR
mean_tpr_cnn = []  # Collect TPR values across folds

for fold in range(5):
    # Load data for the current fold
    X_train_fold, y_train_fold = X_train_folds[fold], y_train_folds[fold]
    X_test_fold, y_test_fold = X_test_folds[fold], y_test_folds[fold]
    
    # Create DataLoaders
    train_loader, test_loader = create_data_loaders(X_train_fold, y_train_fold, X_test_fold, y_test_fold)
    
    # Initialize CNN model
    input_dim = X_train_fold.shape[1]  # Number of features
    output_dim = len(np.unique(y_train_fold))  # Binary or multi-class
    
    cnn_model = CNNModel(input_dim=input_dim, output_dim=output_dim)  # Replace MLPModel with CNNModel
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()  # For classification tasks
    cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
    
    # Train CNN model
    print(f"Training CNN Model for Fold {fold+1}...")
    train_model(cnn_model, train_loader, criterion, cnn_optimizer, num_epochs=5)
    
    # Evaluate CNN model
    accuracy, precision, recall, f1, auc_score, fpr, tpr = evaluate_model(cnn_model, test_loader)
    
    # Store metrics for this fold
    all_accuracies_cnn.append(accuracy)
    all_precisions_cnn.append(precision)
    all_recalls_cnn.append(recall)
    all_f1_scores_cnn.append(f1)
    all_aucs_cnn.append(auc_score)
    
    # Interpolate TPR to mean FPR for consistent evaluation across folds
    tpr_interpolated = np.interp(mean_fpr_cnn, fpr, tpr)
    tpr_interpolated[0] = 0.0  # Ensure the first point is 0 for ROC curve
    mean_tpr_cnn.append(tpr_interpolated)
    
    print(f"CNN Accuracy for Fold {fold+1}: {accuracy:.4f}")
    print(f"CNN Precision for Fold {fold+1}: {precision:.4f}")
    print(f"CNN Recall for Fold {fold+1}: {recall:.4f}")
    print(f"CNN F1-Score for Fold {fold+1}: {f1:.4f}")
    print(f"CNN AUC for Fold {fold+1}: {auc_score:.4f}")

# Finalize mean TPR for CNN model
mean_tpr_cnn = np.array(mean_tpr_cnn)  # Convert to NumPy array
mean_tpr_cnn = np.mean(mean_tpr_cnn, axis=0)  # Compute mean across folds
mean_tpr_cnn[-1] = 1.0  # Ensure the last point is 1 for ROC curve

# Calculate average metrics for CNN model across all folds
print(f"\nAverage CNN Accuracy: {np.mean(all_accuracies_cnn):.4f}")
print(f"Average CNN Precision: {np.mean(all_precisions_cnn):.4f}")
print(f"Average CNN Recall: {np.mean(all_recalls_cnn):.4f}")
print(f"Average CNN F1-Score: {np.mean(all_f1_scores_cnn):.4f}")
print(f"Average CNN AUC: {np.mean(all_aucs_cnn):.4f}")

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Calculate the mean AUC score across all folds
mean_auc_cnn = np.mean(all_aucs_mlp)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(mean_fpr_mlp, mean_tpr_mlp, color='b', label=f'CNN Mean ROC Curve (AUC = {mean_auc_cnn:.4f})')

# Plot a diagonal line for the "no skill" classifier (random guessing)
plt.plot([0, 1], [0, 1], color='grey', linestyle='--', label="No Skill (AUC = 0.5)")

# Add labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('CNN Model ROC Curve')
plt.legend(loc='lower right')

# Show the plot
plt.show()

# MLP Model Training and Evaluation
all_accuracies_mlp, all_precisions_mlp, all_recalls_mlp, all_f1_scores_mlp, all_aucs_mlp = [], [], [], [], []
mean_fpr_mlp = np.linspace(0, 1, 100)  # 100 evenly spaced points for FPR
mean_tpr_mlp = []  # Collect TPR values across folds

for fold in range(5):
    # Load data for the current fold
    X_train_fold, y_train_fold = X_train_folds[fold], y_train_folds[fold]
    X_test_fold, y_test_fold = X_test_folds[fold], y_test_folds[fold]
    
    # Create DataLoaders
    train_loader, test_loader = create_data_loaders(X_train_fold, y_train_fold, X_test_fold, y_test_fold)
    
    # Initialize MLP model
    input_dim = X_train_fold.shape[1]  # Number of features
    output_dim = len(np.unique(y_train_fold))  # Binary or multi-class
    
    mlp_model = MLPModel(input_dim=input_dim, output_dim=output_dim)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()  # For classification tasks
    mlp_optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)
    
    # Train MLP model
    print(f"Training MLP Model for Fold {fold+1}...")
    train_model(mlp_model, train_loader, criterion, mlp_optimizer, num_epochs=5)
    
    # Evaluate MLP model
    accuracy, precision, recall, f1, auc_score, fpr, tpr = evaluate_model(mlp_model, test_loader)
    
    # Store metrics for this fold
    all_accuracies_mlp.append(accuracy)
    all_precisions_mlp.append(precision)
    all_recalls_mlp.append(recall)
    all_f1_scores_mlp.append(f1)
    all_aucs_mlp.append(auc_score)
    
    # Interpolate TPR to mean FPR for consistent evaluation across folds
    tpr_interpolated = np.interp(mean_fpr_mlp, fpr, tpr)
    tpr_interpolated[0] = 0.0  # Ensure the first point is 0 for ROC curve
    mean_tpr_mlp.append(tpr_interpolated)
    
    print(f"MLP Accuracy for Fold {fold+1}: {accuracy:.4f}")
    print(f"MLP Precision for Fold {fold+1}: {precision:.4f}")
    print(f"MLP Recall for Fold {fold+1}: {recall:.4f}")
    print(f"MLP F1-Score for Fold {fold+1}: {f1:.4f}")
    print(f"MLP AUC for Fold {fold+1}: {auc_score:.4f}")

# Finalize mean TPR for MLP model
mean_tpr_mlp = np.array(mean_tpr_mlp)  # Convert to NumPy array
mean_tpr_mlp = np.mean(mean_tpr_mlp, axis=0)  # Compute mean across folds
mean_tpr_mlp[-1] = 1.0  # Ensure the last point is 1 for ROC curve

# Calculate average metrics for MLP model across all folds
print(f"\nAverage MLP Accuracy: {np.mean(all_accuracies_mlp):.4f}")
print(f"Average MLP Precision: {np.mean(all_precisions_mlp):.4f}")
print(f"Average MLP Recall: {np.mean(all_recalls_mlp):.4f}")
print(f"Average MLP F1-Score: {np.mean(all_f1_scores_mlp):.4f}")
print(f"Average MLP AUC: {np.mean(all_aucs_mlp):.4f}")

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Calculate the mean AUC score across all folds
mean_auc_mlp = np.mean(all_aucs_mlp)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(mean_fpr_mlp, mean_tpr_mlp, color='b', label=f'MLP Mean ROC Curve (AUC = {mean_auc_mlp:.4f})')

# Plot a diagonal line for the "no skill" classifier (random guessing)
plt.plot([0, 1], [0, 1], color='grey', linestyle='--', label="No Skill (AUC = 0.5)")

# Add labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('MLP Model ROC Curve')
plt.legend(loc='lower right')

# Show the plot
plt.show()

import pandas as pd

# Store average metrics for CNN and MLP
results = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'],
    'CNN': [
        np.mean(all_accuracies_cnn),
        np.mean(all_precisions_cnn),
        np.mean(all_recalls_cnn),
        np.mean(all_f1_scores_cnn),
        np.mean(all_aucs_cnn)
    ],
    'MLP': [
        np.mean(all_accuracies_mlp),
        np.mean(all_precisions_mlp),
        np.mean(all_recalls_mlp),
        np.mean(all_f1_scores_mlp),
        np.mean(all_aucs_mlp)
    ]
}

# Create a DataFrame to display the results
comparison_df = pd.DataFrame(results)

# Display the comparison DataFrame
print(comparison_df)

import matplotlib.pyplot as plt

# Prepare data for plotting
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
cnn_scores = [np.mean(all_accuracies_cnn), np.mean(all_precisions_cnn), np.mean(all_recalls_cnn),
              np.mean(all_f1_scores_cnn), np.mean(all_aucs_cnn)]
mlp_scores = [np.mean(all_accuracies_mlp), np.mean(all_precisions_mlp), np.mean(all_recalls_mlp),
              np.mean(all_f1_scores_mlp), np.mean(all_aucs_mlp)]

# Plot bar chart for comparison
x = range(len(metrics))
width = 0.35  # width of bars

plt.figure(figsize=(10, 6))

# Bar chart for CNN and MLP
plt.bar(x, cnn_scores, width, label='CNN', color='b', alpha=0.7)
plt.bar([p + width for p in x], mlp_scores, width, label='MLP', color='r', alpha=0.7)

# Add labels, title, and legend
plt.xlabel('Metrics')
plt.ylabel('Scores')
plt.title('Model Comparison: CNN vs MLP')
plt.xticks([p + width / 2 for p in x], metrics)
plt.legend()

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

# Calculate the mean AUC scores for both models
mean_auc_mlp = np.mean(all_aucs_mlp)
mean_auc_cnn = np.mean(all_aucs_cnn)  # Assuming the AUCs for CNN are stored in all_aucs_mlp, adjust as necessary

# Plot ROC curves for both MLP and CNN on the same plot
plt.figure(figsize=(8, 6))

# Plot MLP ROC curve
plt.plot(mean_fpr_mlp, mean_tpr_mlp, color='b', label=f'MLP Mean ROC Curve (AUC = {mean_auc_mlp:.4f})')

# Plot CNN ROC curve
plt.plot(mean_fpr_cnn, mean_tpr_cnn, color='r', label=f'CNN Mean ROC Curve (AUC = {mean_auc_cnn:.4f})')

# Plot a diagonal line for the "no skill" classifier (random guessing)
plt.plot([0, 1], [0, 1], color='grey', linestyle='--', label="No Skill (AUC = 0.5)")

# Add labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('MLP and CNN Model ROC Curves')

# Show the legend
plt.legend(loc='lower right')

# Show the plot
plt.show()