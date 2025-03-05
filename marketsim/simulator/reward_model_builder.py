import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from marketsim.simulator.sampled_arrival_simulator import SimulatorSampledArrival
from marketsim.agent.reward_model_agent import RewardModelAgent
from marketsim.simulator.reward_model_data_collector import RewardModelDataCollector

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.model(x)

def train_nn_model(X_train, y_train, X_val, y_val, batch_size=64, epochs=10, lr=0.001):
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    model = NeuralNetwork(X_train.shape[1])
    
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
            
            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return model, train_losses, val_losses

def evaluate_model(model, X, y, model_type='nn'):
    if model_type == 'nn':
        X_tensor = torch.FloatTensor(X)
        model.eval()
        with torch.no_grad():
            y_pred_proba = model(X_tensor).numpy().flatten()
            y_pred = (y_pred_proba >= 0.5).astype(int)
    else:  # XGBoost
        y_pred_proba = model.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_pred_proba)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

def collect_data(total_samples=1000000, samples_per_sim=5, num_background_agents=24):
    """
    Run multiple simulations to collect the required data.
    """
    data_collector = RewardModelDataCollector()
    
    # Create a progress bar
    pbar = tqdm(total=total_samples, desc="Collecting data")
    
    while data_collector.index < total_samples:
        # Create a new simulator with our reward model agent
        simulator = SimulatorSampledArrival(
            num_background_agents=num_background_agents,
            sim_time=2000,  # Ensure we get enough samples
            num_assets=1,
            lam=0.0005,
            mean=1e5,
            r=0.05,
            shock_var=20000,
            q_max=10,
            pv_var=2e7,
            shade=[250, 500],
            eta=0.2
        )
        
        # Add our reward model agent as the 25th agent
        reward_agent_id = num_background_agents + (1 if 'mm_agent' in simulator.agents else 0) + (1 if 'tron_agent' in simulator.agents else 0)
        simulator.agents[reward_agent_id] = RewardModelAgent(
            agent_id=reward_agent_id,
            market=simulator.markets[0],
            q_max=10,
            pv_var=5e6,
            data_collector=data_collector
        )
        
        # Add initial arrivals for our reward agent - use the same mechanism as ZI agents
        # Take the next arrival time in the simulator's queue
        simulator.arrivals[simulator.arrival_times[simulator.arrival_index].item()].append(reward_agent_id)
        simulator.arrival_index += 1
        
        # Store current index to calculate progress
        prev_index = data_collector.index
        
        # Run the simulation
        simulator.run()
        
        # Update progress bar with the number of new samples collected
        new_samples = data_collector.index - prev_index
        pbar.update(new_samples)
    
    pbar.close()
    print(f"Data collection complete. Collected {data_collector.index} samples.")
    return data_collector.get_dataframe()

def build_reward_models(df):
    """
    Build and evaluate neural network and XGBoost models.
    """
    # Split the data
    X = df.drop('executed', axis=1).values
    y = df['executed'].values
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.22, random_state=42)  # 0.22 of 0.9 = 0.2 of total
    
    print(f"Train size: {X_train.shape[0]}, Validation size: {X_val.shape[0]}, Test size: {X_test.shape[0]}")
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Train Neural Network model
    print("\nTraining Neural Network model...")
    nn_model, train_losses, val_losses = train_nn_model(X_train, y_train, X_val, y_val, 
                                                       batch_size=256, epochs=20, lr=0.001)
    
    # Train XGBoost model
    print("\nTraining XGBoost model...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        random_state=42
    )
    
    # Simple fit without early stopping
    xgb_model.fit(X_train, y_train)
    
    # Manually check validation performance
    val_pred = xgb_model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_pred)
    print(f"XGBoost validation AUC: {val_auc:.4f}")
    
    # Evaluate models
    print("\nEvaluating Neural Network model on test set...")
    nn_metrics = evaluate_model(nn_model, X_test, y_test, model_type='nn')
    
    print("\nEvaluating XGBoost model on test set...")
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, model_type='xgb')
    
    # Print results
    print("\nNeural Network metrics:")
    for metric, value in nn_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nXGBoost metrics:")
    for metric, value in xgb_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot training history for NN
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Neural Network Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('nn_training_history.png')
    
    # Plot feature importance for XGBoost
    feature_names = df.drop('executed', axis=1).columns
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(xgb_model, max_num_features=len(feature_names))
    plt.title('XGBoost Feature Importance')
    plt.savefig('xgb_feature_importance.png')
    
    return nn_model, xgb_model, nn_metrics, xgb_metrics

if __name__ == "__main__":
    # Number of samples to collect
    total_samples = 1000000
    
    print("Collecting data from simulations...")
    df = collect_data(total_samples=total_samples)
    
    # Save the raw data
    df.to_csv("reward_model_data.csv", index=False)
    print(f"Collected {len(df)} samples. Data saved to 'reward_model_data.csv'")
    
    print("\nBuilding reward models...")
    nn_model, xgb_model, nn_metrics, xgb_metrics = build_reward_models(df)
    
    # Save the models
    torch.save(nn_model.state_dict(), "nn_reward_model.pt")
    xgb_model.save_model("xgb_reward_model.json")
    
    print("\nModels saved to 'nn_reward_model.pt' and 'xgb_reward_model.json'") 