import numpy as np
import pandas as pd
import joblib
import json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns

print("""
╔══════════════════════════════════════════════════════════╗
║              MODEL TRAINING & COMPARISON                ║
║                                                          ║
║  Models:                                                 ║
║  1. Logistic Regression (Baseline)                      ║
║  2. Random Forest                                        ║
║  3. XGBoost                                             ║
║  4. Neural Network                                       ║
╚══════════════════════════════════════════════════════════╝
""")

# ============================================================
# 1. LOAD PROCESSED DATA
# ============================================================

def load_processed_data():
    """Load preprocessed datasets"""
    print("\n" + "="*60)
    print("LOADING PROCESSED DATA")
    print("="*60)
    
    X_train = np.load("data/X_train.npy")
    X_val = np.load("data/X_val.npy")
    X_test = np.load("data/X_test.npy")
    y_train = np.load("data/y_train.npy")
    y_val = np.load("data/y_val.npy")
    y_test = np.load("data/y_test.npy")
    
    with open("data/feature_names.json", "r") as f:
        feature_names = json.load(f)
    
    label_encoder = joblib.load("models/label_encoder.pkl")
    
    print(f"✅ Training set:   {X_train.shape}")
    print(f"✅ Validation set: {X_val.shape}")
    print(f"✅ Test set:       {X_test.shape}")
    print(f"✅ Features:       {len(feature_names)}")
    print(f"✅ Classes:        {len(label_encoder.classes_)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names, label_encoder


# ============================================================
# 2. TRAIN LOGISTIC REGRESSION (BASELINE)
# ============================================================

def train_logistic_regression(X_train, y_train, X_val, y_val):
    """Train Logistic Regression baseline model"""
    print("\n" + "="*60)
    print("MODEL 1: LOGISTIC REGRESSION")
    print("="*60)
    
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        multi_class='multinomial',
        solver='lbfgs'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_acc = model.score(X_train, y_train)
    val_acc = model.score(X_val, y_val)
    
    print(f"✅ Training Accuracy:   {train_acc:.4f}")
    print(f"✅ Validation Accuracy: {val_acc:.4f}")
    
    # Save model
    joblib.dump(model, "models/logistic_regression.pkl")
    print(f"💾 Saved to: models/logistic_regression.pkl")
    
    return model, {"train_acc": train_acc, "val_acc": val_acc}


# ============================================================
# 3. TRAIN RANDOM FOREST
# ============================================================

def train_random_forest(X_train, y_train, X_val, y_val):
    """Train Random Forest model"""
    print("\n" + "="*60)
    print("MODEL 2: RANDOM FOREST")
    print("="*60)
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_acc = model.score(X_train, y_train)
    val_acc = model.score(X_val, y_val)
    
    print(f"✅ Training Accuracy:   {train_acc:.4f}")
    print(f"✅ Validation Accuracy: {val_acc:.4f}")
    
    # Save model
    joblib.dump(model, "models/random_forest.pkl")
    print(f"💾 Saved to: models/random_forest.pkl")
    
    return model, {"train_acc": train_acc, "val_acc": val_acc}


# ============================================================
# 4. TRAIN XGBOOST
# ============================================================

def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost model"""
    print("\n" + "="*60)
    print("MODEL 3: XGBOOST")
    print("="*60)
    
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss'
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Evaluate
    train_acc = model.score(X_train, y_train)
    val_acc = model.score(X_val, y_val)
    
    print(f"✅ Training Accuracy:   {train_acc:.4f}")
    print(f"✅ Validation Accuracy: {val_acc:.4f}")
    
    # Save model
    joblib.dump(model, "models/xgboost.pkl")
    print(f"💾 Saved to: models/xgboost.pkl")
    
    return model, {"train_acc": train_acc, "val_acc": val_acc}


# ============================================================
# 5. TRAIN NEURAL NETWORK
# ============================================================

def train_neural_network(X_train, y_train, X_val, y_val, num_classes):
    """Train Neural Network model"""
    print("\n" + "="*60)
    print("MODEL 4: NEURAL NETWORK")
    print("="*60)
    
    # Convert labels to categorical
    y_train_cat = keras.utils.to_categorical(y_train, num_classes)
    y_val_cat = keras.utils.to_categorical(y_val, num_classes)
    
    # Build model
    model = keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n📊 Model Architecture:")
    model.summary()
    
    # Train
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=50,
        batch_size=32,
        verbose=0
    )
    
    # Evaluate
    train_loss, train_acc = model.evaluate(X_train, y_train_cat, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val_cat, verbose=0)
    
    print(f"\n✅ Training Accuracy:   {train_acc:.4f}")
    print(f"✅ Validation Accuracy: {val_acc:.4f}")
    
    # Save model
    model.save("models/neural_network.h5")
    print(f"💾 Saved to: models/neural_network.h5")
    
    return model, {"train_acc": train_acc, "val_acc": val_acc, "history": history}


# ============================================================
# 6. EVALUATE ON TEST SET
# ============================================================

def evaluate_model(model, X_test, y_test, label_encoder, model_name):
    """Evaluate model on test set"""
    print(f"\n{'='*60}")
    print(f"EVALUATING {model_name} ON TEST SET")
    print("="*60)
    
    # Handle neural network separately
    if model_name == "Neural Network":
        y_test_cat = keras.utils.to_categorical(y_test, len(label_encoder.classes_))
        test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    else:
        test_acc = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
    
    print(f"✅ Test Accuracy: {test_acc:.4f}")
    
    # Classification report
    print(f"\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return test_acc, y_pred, cm


# ============================================================
# 7. COMPARE ALL MODELS
# ============================================================

def compare_models(results, label_encoder):
    """Create comparison table of all models"""
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    
    comparison_df = pd.DataFrame(results).T
    comparison_df.columns = ["Train Acc", "Val Acc", "Test Acc"]
    comparison_df = comparison_df.sort_values("Test Acc", ascending=False)
    
    print("\n" + str(comparison_df))
    
    # Save comparison
    comparison_df.to_csv("results/model_comparison.csv")
    print(f"\n💾 Saved comparison to: results/model_comparison.csv")
    
    # Determine best model
    best_model_name = comparison_df["Test Acc"].idxmax()
    best_test_acc = comparison_df.loc[best_model_name, "Test Acc"]
    
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"   Test Accuracy: {best_test_acc:.4f}")
    
    # Save best model name
    with open("models/best_model.txt", "w") as f:
        f.write(best_model_name)
    
    return best_model_name


# ============================================================
# 8. MAIN TRAINING PIPELINE
# ============================================================

def main():
    import os
    os.makedirs("results", exist_ok=True)
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names, label_encoder = load_processed_data()
    
    results = {}
    
    # Train Logistic Regression
    lr_model, lr_metrics = train_logistic_regression(X_train, y_train, X_val, y_val)
    lr_test_acc, _, _ = evaluate_model(lr_model, X_test, y_test, label_encoder, "Logistic Regression")
    results["Logistic Regression"] = [lr_metrics["train_acc"], lr_metrics["val_acc"], lr_test_acc]
    
    # Train Random Forest
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_val, y_val)
    rf_test_acc, _, _ = evaluate_model(rf_model, X_test, y_test, label_encoder, "Random Forest")
    results["Random Forest"] = [rf_metrics["train_acc"], rf_metrics["val_acc"], rf_test_acc]
    
    # Train XGBoost
    xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_val, y_val)
    xgb_test_acc, _, _ = evaluate_model(xgb_model, X_test, y_test, label_encoder, "XGBoost")
    results["XGBoost"] = [xgb_metrics["train_acc"], xgb_metrics["val_acc"], xgb_test_acc]
    
    # Train Neural Network
    nn_model, nn_metrics = train_neural_network(X_train, y_train, X_val, y_val, len(label_encoder.classes_))
    nn_test_acc, _, _ = evaluate_model(nn_model, X_test, y_test, label_encoder, "Neural Network")
    results["Neural Network"] = [nn_metrics["train_acc"], nn_metrics["val_acc"], nn_test_acc]
    
    # Compare all models
    best_model = compare_models(results, label_encoder)
    
    print("\n✅ Model training complete!")
    print(f"🏆 Best model: {best_model}")


if __name__ == "__main__":
    main()