#!/usr/bin/env python3
"""
Train handwriting-based Parkinson's Disease detection model.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

def print_section(title):
    """Print formatted section header."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")

def load_data():
    """Load handwriting dataset."""
    data_path = Path('data/raw/handwriting/handwriting_data.csv')
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Handwriting dataset not found at {data_path}. "
            "Run generate_modality_datasets.py first."
        )
    
    df = pd.read_csv(data_path)
    
    # Separate features and labels
    feature_cols = [col for col in df.columns if col not in ['sample_id', 'status']]
    X = df[feature_cols].values
    y = df['status'].values
    
    return X, y, feature_cols

def train_model(X_train, y_train, X_val, y_val):
    """Train and optimize handwriting model."""
    print_section("TRAINING HANDWRITING MODEL")
    
    # 1. SVM with RBF kernel (proven effective for PD detection)
    print("\n1. Training SVM with GridSearch...")
    svm_params = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01],
        'kernel': ['rbf']
    }
    
    svm = GridSearchCV(
        SVC(probability=True, random_state=42),
        svm_params,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    svm.fit(X_train, y_train)
    print(f"   Best SVM params: {svm.best_params_}")
    print(f"   Best CV score: {svm.best_score_:.4f}")
    
    # 2. Random Forest
    print("\n2. Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_score = cross_val_score(rf, X_train, y_train, cv=5, scoring='f1').mean()
    print(f"   RF CV score: {rf_score:.4f}")
    
    # 3. Voting Ensemble
    print("\n3. Creating Voting Ensemble...")
    ensemble = VotingClassifier(
        estimators=[
            ('svm', svm.best_estimator_),
            ('rf', rf)
        ],
        voting='soft',
        weights=[1.5, 1.0]  # SVM gets higher weight
    )
    ensemble.fit(X_train, y_train)
    
    # Evaluate on validation set
    models = {
        'SVM': svm.best_estimator_,
        'Random Forest': rf,
        'Ensemble': ensemble
    }
    
    best_model = None
    best_score = 0
    
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    
    for name, model in models.items():
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred)
        rec = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_proba)
        
        print(f"\n{name}:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {auc:.4f}")
        
        if f1 > best_score:
            best_score = f1
            best_model = (name, model)
    
    print(f"\n✓ Best model: {best_model[0]} (F1={best_score:.4f})")
    return best_model[1], best_model[0]

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model on test set."""
    print_section("TEST SET EVALUATION")
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba[:, 1])
    
    print(f"\n{model_name} - Test Results:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {auc:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Healthy', 'PD']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Save plots
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    # Confusion Matrix Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Healthy', 'PD'],
                yticklabels=['Healthy', 'PD'])
    plt.title(f'Handwriting Model - Confusion Matrix\n{model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(models_dir / 'handwriting_confusion_matrix.png', dpi=150)
    plt.close()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Handwriting Model - ROC Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(models_dir / 'handwriting_roc_curve.png', dpi=150)
    plt.close()
    
    print(f"\n✓ Plots saved to {models_dir}/")
    
    return {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1),
        'roc_auc': float(auc),
        'confusion_matrix': cm.tolist()
    }

def main():
    print_section("HANDWRITING-BASED PD DETECTION MODEL TRAINING")
    
    # Load data
    print("\n1. Loading handwriting dataset...")
    X, y, feature_names = load_data()
    print(f"   ✓ Loaded {len(X)} samples with {X.shape[1]} features")
    print(f"   - Healthy: {(y==0).sum()}")
    print(f"   - PD: {(y==1).sum()}")
    
    # Split data
    print("\n2. Splitting data...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    print(f"   Train: {len(X_train)} samples")
    print(f"   Val:   {len(X_val)} samples")
    print(f"   Test:  {len(X_test)} samples")
    
    # Scale features
    print("\n3. Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    print("   ✓ Features scaled using StandardScaler")
    
    # Train model
    model, model_name = train_model(X_train_scaled, y_train, X_val_scaled, y_val)
    
    # Evaluate on test set
    metrics = evaluate_model(model, X_test_scaled, y_test, model_name)
    
    # Save model
    print_section("SAVING MODEL")
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / 'handwriting_model.joblib'
    scaler_path = models_dir / 'handwriting_scaler.joblib'
    
    # Save with metadata
    model_data = {
        'model': model,
        'model_name': model_name,
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'test_metrics': metrics
    }
    
    joblib.dump(model_data, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"\n✓ Model saved to: {model_path}")
    print(f"✓ Scaler saved to: {scaler_path}")
    
    print_section("TRAINING COMPLETE!")
    print(f"\nHandwriting Model Summary:")
    print(f"  Model Type: {model_name}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Test F1-Score: {metrics['f1_score']:.4f}")
    print(f"  Test ROC-AUC: {metrics['roc_auc']:.4f}")

if __name__ == '__main__':
    main()

