import numpy as np
import pandas as pd
import joblib
import json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    f1_score, roc_auc_score, make_scorer
)
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.base import clone
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from scipy import stats

# Add project root to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config

print("""
╔══════════════════════════════════════════════════════════╗
║       ENHANCED MODEL TRAINING WITH ADVANCED ML           ║
║                                                          ║
║  ✨ New Features:                                        ║
║  • Stratified K-Fold Cross-Validation                    ║
║  • Class Imbalance Handling (SMOTE)                      ║
║  • Feature Selection (RFE)                               ║
║  • Model Calibration                                     ║
║  • Ensemble Voting Classifier                            ║
║  • ROC-AUC & Detailed Metrics                            ║
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
    
    X_train = np.load(config.X_TRAIN)
    X_val = np.load(config.X_VAL)
    X_test = np.load(config.X_TEST)
    y_train = np.load(config.Y_TRAIN)
    y_val = np.load(config.Y_VAL)
    y_test = np.load(config.Y_TEST)
    
    with open(config.FEATURE_NAMES_JSON, "r") as f:
        feature_names = json.load(f)
    
    label_encoder = joblib.load(config.LABEL_ENCODER)
    
    print(f"✅ Training set:   {X_train.shape}")
    print(f"✅ Validation set: {X_val.shape}")
    print(f"✅ Test set:       {X_test.shape}")
    print(f"✅ Features:       {len(feature_names)}")
    print(f"✅ Classes:        {len(label_encoder.classes_)}")
    
    # Class distribution
    print(f"\n📊 Class Distribution (Training):")
    unique, counts = np.unique(y_train, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"   {label_encoder.classes_[cls]}: {count} ({count/len(y_train)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names, label_encoder


# ============================================================
# 2. HANDLE CLASS IMBALANCE WITH SMOTE
# ============================================================

def balance_dataset(X_train, y_train, method='smote'):
    """Balance dataset using SMOTE or SMOTETomek"""
    print("\n" + "="*60)
    print("HANDLING CLASS IMBALANCE")
    print("="*60)
    
    print(f"Original training size: {X_train.shape[0]}")
    
    if method == 'smote':
        sampler = SMOTE(random_state=42, k_neighbors=5)
        print("Using SMOTE (Synthetic Minority Over-sampling)")
    elif method == 'smote_tomek':
        sampler = SMOTETomek(random_state=42)
        print("Using SMOTETomek (Over-sampling + Under-sampling)")
    else:
        print("No balancing applied")
        return X_train, y_train
    
    X_balanced, y_balanced = sampler.fit_resample(X_train, y_train)
    
    print(f"Balanced training size: {X_balanced.shape[0]}")
    print(f"\n📊 New Class Distribution:")
    unique, counts = np.unique(y_balanced, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"   Class {cls}: {count} ({count/len(y_balanced)*100:.1f}%)")
    
    return X_balanced, y_balanced


# ============================================================
# 3. FEATURE SELECTION WITH RFE
# ============================================================

def select_features(X_train, y_train, X_val, X_test, feature_names, n_features='auto'):
    """Select best features using Recursive Feature Elimination"""
    print("\n" + "="*60)
    print("FEATURE SELECTION (RFE)")
    print("="*60)
    
    # Use Random Forest for feature importance
    estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    if n_features == 'auto':
        # Automatic selection using cross-validation
        selector = RFECV(
            estimator=estimator,
            step=1,
            cv=StratifiedKFold(5),
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
    else:
        from sklearn.feature_selection import RFE
        selector = RFE(estimator=estimator, n_features_to_select=n_features)
    
    print("🔍 Selecting optimal features...")
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_val_selected = selector.transform(X_val)
    X_test_selected = selector.transform(X_test)
    
    # Get selected feature names
    selected_indices = selector.get_support(indices=True)
    selected_features = [feature_names[i] for i in selected_indices]
    
    print(f"\n✅ Selected {X_train_selected.shape[1]} features out of {X_train.shape[1]}")
    print(f"📊 Selected features: {', '.join(selected_features[:10])}...")
    
    return X_train_selected, X_val_selected, X_test_selected, selected_features


# ============================================================
# 4. STRATIFIED K-FOLD CROSS-VALIDATION
# ============================================================

def evaluate_with_cv(model, X, y, model_name, cv=5):
    """Evaluate model using Stratified K-Fold CV"""
    print(f"\n🔄 {model_name}: Stratified {cv}-Fold Cross-Validation...")
    
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Multiple scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'f1_macro': 'f1_macro',
        'f1_weighted': 'f1_weighted'
    }
    
    scores = {}
    for metric_name, metric in scoring.items():
        cv_scores = cross_val_score(model, X, y, cv=skf, scoring=metric, n_jobs=-1)
        scores[metric_name] = {
            'mean': cv_scores.mean(),
            'std': cv_scores.std(),
            'scores': cv_scores
        }
        print(f"   {metric_name}: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    return scores


# ============================================================
# 5. TRAIN LOGISTIC REGRESSION (ENHANCED)
# ============================================================

def train_logistic_regression(X_train, y_train, X_val, y_val):
    """Train Logistic Regression with Stratified CV"""
    print("\n" + "="*60)
    print("MODEL 1: LOGISTIC REGRESSION (Enhanced)")
    print("="*60)
    
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'saga'],
        'max_iter': [5000],
        'class_weight': ['balanced', None]  # Handle class imbalance
    }
    
    base_model = LogisticRegression(
        random_state=42
    )
    
    # Use Stratified K-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("🔍 GridSearch with Stratified K-Fold CV...")
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=skf,
        scoring='f1_macro',  # Better for imbalanced data
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_
    
    print(f"\n✅ Best parameters: {grid_search.best_params_}")
    print(f"✅ Best CV F1-Macro: {grid_search.best_score_:.4f}")
    
    # Calibrate probabilities
    print("🎯 Calibrating model probabilities...")
    calibrated_model = CalibratedClassifierCV(model, cv=3, method='sigmoid')
    calibrated_model.fit(X_train, y_train)
    
    # Evaluate
    train_acc = calibrated_model.score(X_train, y_train)
    val_acc = calibrated_model.score(X_val, y_val)
    
    print(f"✅ Training Accuracy:   {train_acc:.4f}")
    print(f"✅ Validation Accuracy: {val_acc:.4f}")
    
    joblib.dump(calibrated_model, config.LOGISTIC_REGRESSION_MODEL)
    print(f"💾 Saved calibrated model")
    
    return calibrated_model, {
        "train_acc": train_acc, 
        "val_acc": val_acc, 
        "cv_f1": grid_search.best_score_
    }


# ============================================================
# 6. TRAIN RANDOM FOREST (ENHANCED)
# ============================================================

def train_random_forest(X_train, y_train, X_val, y_val):
    """Train Random Forest with enhanced parameters"""
    print("\n" + "="*60)
    print("MODEL 2: RANDOM FOREST (Enhanced)")
    print("="*60)
    
    param_distributions = {
        'n_estimators': [200, 300, 500],
        'max_depth': [15, 20, 25, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.5],
        'bootstrap': [True],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }
    
    base_model = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        oob_score=True  # Out-of-bag score
    )
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("🔍 RandomizedSearch with Stratified K-Fold CV...")
    random_search = RandomizedSearchCV(
        base_model,
        param_distributions,
        n_iter=100,  # More iterations
        cv=skf,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    random_search.fit(X_train, y_train)
    model = random_search.best_estimator_
    
    print(f"\n✅ Best parameters: {random_search.best_params_}")
    print(f"✅ Best CV F1-Macro: {random_search.best_score_:.4f}")
    print(f"✅ OOB Score: {model.oob_score_:.4f}")
    
    train_acc = model.score(X_train, y_train)
    val_acc = model.score(X_val, y_val)
    
    print(f"✅ Training Accuracy:   {train_acc:.4f}")
    print(f"✅ Validation Accuracy: {val_acc:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': range(X_train.shape[1]),
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n📊 Top 10 Important Features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"   Feature {int(row['feature'])}: {row['importance']:.4f}")
    
    joblib.dump(model, config.RANDOM_FOREST_MODEL)
    
    return model, {
        "train_acc": train_acc, 
        "val_acc": val_acc, 
        "cv_f1": random_search.best_score_,
        "oob_score": model.oob_score_
    }


# ============================================================
# 7. TRAIN XGBOOST (ENHANCED)
# ============================================================

def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost with enhanced parameters"""
    print("\n" + "="*60)
    print("MODEL 3: XGBOOST (Enhanced)")
    print("="*60)
    
    # Calculate class weights
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights = compute_sample_weight('balanced', y_train)
    
    param_grid = {
        'n_estimators': [200, 300, 500],
        'max_depth': [6, 8, 10, 12],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2],
        'min_child_weight': [1, 3, 5]
        # scale_pos_weight removed as it's for binary classification
    }
    
    base_model = xgb.XGBClassifier(
        random_state=42,
        eval_metric='mlogloss',
        early_stopping_rounds=20,
        device='cuda',  # Enable GPU
        tree_method='hist' # Optimized for GPU
    )
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("🔍 RandomizedSearch (50 iterations)...")
    random_search = RandomizedSearchCV(
        base_model,
        param_grid,
        n_iter=50,
        cv=skf,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    random_search.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    model = random_search.best_estimator_
    
    print(f"\n✅ Best parameters: {random_search.best_params_}")
    print(f"✅ Best CV F1-Macro: {random_search.best_score_:.4f}")
    
    train_acc = model.score(X_train, y_train)
    val_acc = model.score(X_val, y_val)
    
    print(f"✅ Training Accuracy:   {train_acc:.4f}")
    print(f"✅ Validation Accuracy: {val_acc:.4f}")
    
    joblib.dump(model, config.XGBOOST_MODEL)
    
    return model, {
        "train_acc": train_acc, 
        "val_acc": val_acc, 
        "cv_f1": random_search.best_score_
    }


# ============================================================
# 8. TRAIN NEURAL NETWORK (GPU ENABLED)
# ============================================================

def train_neural_network(X_train, y_train, X_val, y_val, input_dim, num_classes):
    """Train Neural Network using TensorFlow/Keras (GPU)"""
    print("\n" + "="*60)
    print("MODEL 4: NEURAL NETWORK (GPU Enabled)")
    print("="*60)
    
    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ TensorFlow using GPU: {gpus[0]}")
    else:
        print("⚠️  TensorFlow using CPU (No GPU detected)")
    
    # Convert labels to categorical
    y_train_cat = keras.utils.to_categorical(y_train, num_classes)
    y_val_cat = keras.utils.to_categorical(y_val, num_classes)
    
    # Define model architecture
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
    ]
    
    print("🔄 Training Neural Network...")
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    val_loss, val_acc = model.evaluate(X_val, y_val_cat, verbose=0)
    print(f"✅ Validation Accuracy: {val_acc:.4f}")
    
    # Save model
    model.save(config.NEURAL_NETWORK_MODEL)
    print(f"💾 Saved Neural Network model")
    
    return model, {"train_acc": history.history['accuracy'][-1], "val_acc": val_acc}


# ============================================================
# 9. TRAIN ENSEMBLE VOTING CLASSIFIER
# ============================================================

def train_ensemble(lr_model, rf_model, xgb_model, nn_model, X_train, y_train, X_val, y_val):
    """Create ensemble of best models"""
    print("\n" + "="*60)
    print("MODEL 5: ENSEMBLE VOTING CLASSIFIER")
    print("="*60)
    
    # Clone XGBoost model and disable early stopping for ensemble
    # VotingClassifier doesn't support eval_set for early stopping easily
    xgb_no_early_stop = clone(xgb_model)
    xgb_no_early_stop.set_params(early_stopping_rounds=None)
    
    ensemble = VotingClassifier(
        estimators=[
            ('lr', lr_model),
            ('rf', rf_model),
            ('xgb', xgb_no_early_stop)
            # Note: Keras model cannot be directly added to sklearn VotingClassifier
            # We would need a wrapper, but for now we'll skip adding NN to this specific ensemble
            # or implement a custom voting mechanism.
            # For simplicity in this step, we keep sklearn models here.
        ],
        voting='soft',  # Use probability averaging
        n_jobs=1  # Avoid contention with already parallelized models
    )
    
    print("🔄 Training ensemble...")
    ensemble.fit(X_train, y_train)
    
    train_acc = ensemble.score(X_train, y_train)
    val_acc = ensemble.score(X_val, y_val)
    
    print(f"✅ Training Accuracy:   {train_acc:.4f}")
    print(f"✅ Validation Accuracy: {val_acc:.4f}")
    
    ensemble_path = os.path.join(config.MODELS_DIR, "ensemble_voting.pkl")
    joblib.dump(ensemble, ensemble_path)
    print(f"💾 Saved ensemble model")
    
    return ensemble, {"train_acc": train_acc, "val_acc": val_acc}


# ============================================================
# 10. ENHANCED EVALUATION WITH ROC-AUC
# ============================================================

def evaluate_model_enhanced(model, X_test, y_test, label_encoder, model_name):
    """Enhanced evaluation with ROC-AUC and detailed metrics"""
    print(f"\n{'='*60}")
    print(f"EVALUATING {model_name} ON TEST SET")
    print("="*60)
    
    # Predictions
    if model_name == "Neural Network":
        y_test_cat = keras.utils.to_categorical(y_test, len(label_encoder.classes_))
        test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        y_pred_proba = model.predict(X_test, verbose=0)
    else:
        test_acc = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    # ROC-AUC (One-vs-Rest)
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
        print(f"✅ Test Accuracy:       {test_acc:.4f}")
        print(f"✅ F1-Score (Macro):    {f1_macro:.4f}")
        print(f"✅ F1-Score (Weighted): {f1_weighted:.4f}")
        print(f"✅ ROC-AUC (Macro):     {roc_auc:.4f}")
    except:
        roc_auc = None
        print(f"✅ Test Accuracy:       {test_acc:.4f}")
        print(f"✅ F1-Score (Macro):    {f1_macro:.4f}")
        print(f"✅ F1-Score (Weighted): {f1_weighted:.4f}")
    
    # Per-class metrics
    print(f"\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📊 Confusion Matrix:")
    cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)
    print(cm_df)
    
    return test_acc, f1_macro, f1_weighted, roc_auc, y_pred, cm


# ============================================================
# 11. MAIN TRAINING PIPELINE
# ============================================================

def main():
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names, label_encoder = load_processed_data()
    
    # Combine train + val for final training
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])
    
    # Optional: Balance dataset with SMOTE
    use_smote = input("\n⚖️  Apply SMOTE to balance classes? [y/N]: ").lower() == 'y'
    if use_smote:
        X_train_full, y_train_full = balance_dataset(X_train_full, y_train_full, method='smote')
    
    # Optional: Feature selection
    use_feature_selection = input("🔍 Apply feature selection (RFE)? [y/N]: ").lower() == 'y'
    if use_feature_selection:
        X_train_full, X_val_fs, X_test, selected_features = select_features(
            X_train_full, y_train_full, X_val, X_test, feature_names
        )
    
    results = {}
    
    print("\n" + "="*70)
    print(" "*15 + "TRAINING ENHANCED MODELS")
    print("="*70)
    
    # Train models
    lr_model, lr_metrics = train_logistic_regression(X_train_full, y_train_full, X_test, y_test)
    lr_test_acc, lr_f1_macro, lr_f1_weighted, lr_roc_auc, _, _ = evaluate_model_enhanced(
        lr_model, X_test, y_test, label_encoder, "Logistic Regression"
    )
    results["Logistic Regression"] = {
        "Train Acc": lr_metrics["train_acc"],
        "Val Acc": lr_metrics["val_acc"],
        "Test Acc": lr_test_acc,
        "F1 Macro": lr_f1_macro,
        "F1 Weighted": lr_f1_weighted,
        "ROC-AUC": lr_roc_auc
    }
    
    rf_model, rf_metrics = train_random_forest(X_train_full, y_train_full, X_test, y_test)
    rf_test_acc, rf_f1_macro, rf_f1_weighted, rf_roc_auc, _, _ = evaluate_model_enhanced(
        rf_model, X_test, y_test, label_encoder, "Random Forest"
    )
    results["Random Forest"] = {
        "Train Acc": rf_metrics["train_acc"],
        "Val Acc": rf_metrics["val_acc"],
        "Test Acc": rf_test_acc,
        "F1 Macro": rf_f1_macro,
        "F1 Weighted": rf_f1_weighted,
        "ROC-AUC": rf_roc_auc
    }
    
    xgb_model, xgb_metrics = train_xgboost(X_train_full, y_train_full, X_test, y_test)
    xgb_test_acc, xgb_f1_macro, xgb_f1_weighted, xgb_roc_auc, _, _ = evaluate_model_enhanced(
        xgb_model, X_test, y_test, label_encoder, "XGBoost"
    )
    results["XGBoost"] = {
        "Train Acc": xgb_metrics["train_acc"],
        "Val Acc": xgb_metrics["val_acc"],
        "Test Acc": xgb_test_acc,
        "F1 Macro": xgb_f1_macro,
        "F1 Weighted": xgb_f1_weighted,
        "ROC-AUC": xgb_roc_auc
    }
    
    # Train Neural Network
    nn_model, nn_metrics = train_neural_network(
        X_train_full, y_train_full, X_test, y_test, 
        input_dim=X_train_full.shape[1], 
        num_classes=len(label_encoder.classes_)
    )
    nn_test_acc, nn_f1_macro, nn_f1_weighted, nn_roc_auc, _, _ = evaluate_model_enhanced(
        nn_model, X_test, y_test, label_encoder, "Neural Network"
    )
    results["Neural Network"] = {
        "Train Acc": nn_metrics["train_acc"],
        "Val Acc": nn_metrics["val_acc"],
        "Test Acc": nn_test_acc,
        "F1 Macro": nn_f1_macro,
        "F1 Weighted": nn_f1_weighted,
        "ROC-AUC": nn_roc_auc
    }
    
    # Train ensemble
    ensemble_model, ensemble_metrics = train_ensemble(
        lr_model, rf_model, xgb_model, nn_model, X_train_full, y_train_full, X_test, y_test
    )
    ens_test_acc, ens_f1_macro, ens_f1_weighted, ens_roc_auc, _, _ = evaluate_model_enhanced(
        ensemble_model, X_test, y_test, label_encoder, "Ensemble Voting"
    )
    results["Ensemble Voting"] = {
        "Train Acc": ensemble_metrics["train_acc"],
        "Val Acc": ensemble_metrics["val_acc"],
        "Test Acc": ens_test_acc,
        "F1 Macro": ens_f1_macro,
        "F1 Weighted": ens_f1_weighted,
        "ROC-AUC": ens_roc_auc
    }
    
    # Compare models
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    
    comparison_df = pd.DataFrame(results).T
    comparison_df = comparison_df.sort_values("Test Acc", ascending=False)
    print("\n" + str(comparison_df))
    
    comparison_df.to_csv(config.MODEL_COMPARISON_CSV)
    
    best_model_name = comparison_df["Test Acc"].idxmax()
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"   Test Accuracy: {comparison_df.loc[best_model_name, 'Test Acc']:.4f}")
    print(f"   F1-Macro:      {comparison_df.loc[best_model_name, 'F1 Macro']:.4f}")
    
    with open(config.BEST_MODEL_TXT, "w") as f:
        f.write(best_model_name)
    
    print("\n✅ TRAINING COMPLETE!")


if __name__ == "__main__":
    main()