#!/usr/bin/env python3
"""
Clinical Progression Evaluation Functions

Implements GATher-style threshold-based binary classification metrics.
"""

import torch
import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix
)
from scipy.stats import fisher_exact
from typing import Dict, List, Tuple


def compute_prediction_variables(predictions: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    """
    Compute 4 prediction variables from model outputs.
    
    Variables:
    1. Positive Efficacy: Direct positive outcome score
    2. Unmet Efficacy: Direct unmet/failure score  
    3. Efficacy Distance: pos - unmet (margin)
    4. Efficacy Ratio: pos / (unmet + ε)
    
    Args:
        predictions: Dict with keys ['pos', 'unmet', 'adv', 'op']
        
    Returns:
        Dict of variable_name -> numpy array of scores
    """
    pos = predictions['pos'].cpu().numpy()
    unmet = predictions['unmet'].cpu().numpy()
    
    return {
        'Positive Efficacy': pos,
        'Unmet Efficacy': unmet,
        'Efficacy Distance': pos - unmet,
        'Efficacy Ratio': pos / (unmet + 1e-6)
    }


def find_best_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    thresholds: List[float] = None
) -> Dict:
    """
    Grid search over thresholds to find best F1 score.
    
    Args:
        scores: Continuous prediction scores
        labels: Binary ground truth (0/1)
        thresholds: List of candidate thresholds to test
        
    Returns:
        Dict with best threshold and corresponding metrics
    """
    if thresholds is None:
        # Default GATher thresholds (around Phase 2 boundary)
        thresholds = [0.5, 1.0, 1.5, 2.0, 2.37, 2.5, 3.0, 3.12, 3.5, 4.0]
    
    best_f1 = 0
    best_result = {}
    
    for thresh in thresholds:
        preds_binary = (scores >= thresh).astype(int)
        
        # Skip if all same class
        if len(np.unique(preds_binary)) < 2:
            continue
        
        # Compute classification metrics
        prec, rec, f1, _ = precision_recall_fscore_support(
            labels, preds_binary, average='binary', zero_division=0
        )
        mcc = matthews_corrcoef(labels, preds_binary)
        
        # ROC AUC (use continuous scores, not thresholded)
        try:
            auc = roc_auc_score(labels, scores)
        except ValueError:
            auc = 0.5  # Fallback if only one class
        
        # Fisher's exact test for statistical significance
        cm = confusion_matrix(labels, preds_binary)
        if cm.shape == (2, 2):
            _, p_val = fisher_exact(cm)
        else:
            p_val = 1.0
        
        # Track best F1
        if f1 > best_f1:
            best_f1 = f1
            best_result = {
                'threshold': thresh,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'mcc': mcc,
                'roc_auc': auc,
                'p_val': p_val,
                'counts': f"{int(labels.sum())}-{int(len(labels) - labels.sum())}"
            }
    
    return best_result if best_result else {
        'threshold': thresholds[0], 'precision': 0, 'recall': 0, 
        'f1': 0, 'mcc': 0, 'roc_auc': 0.5, 'p_val': 1.0, 'counts': '0-0'
    }


def evaluate_clinical_progression(
    model,
    embeddings: Dict[str, torch.Tensor],
    cohort_a_labels: pd.DataFrame,
    cohort_b_labels: pd.DataFrame,
    device,
    thresholds: List[float] = None
) -> Dict:
    """
    Evaluate model using GATher-style clinical progression metrics.
    
    Args:
        model: Trained decoder model
        embeddings: Dict of node embeddings
        cohort_a_labels: Progression labels (disease_idx, target_idx, progressed)
        cohort_b_labels: Regression labels (disease_idx, target_idx, is_regression)
        device: Torch device
        thresholds: List of decision thresholds to test
        
    Returns:
        Dict with results for both cohorts
    """
    print("\n" + "="*80)
    print("📊 Clinical Progression Evaluation (GATher-Style)")
    print("="*80)
    
    model.eval()
    results = {'cohort_a': {}, 'cohort_b': {}}
    
    # --- Cohort A: Positive Progression ---
    print("\n🔬 Cohort A: Positive Progression (First-in-Class Detection)")
    print(f"   Total pairs: {len(cohort_a_labels):,}")
    print(f"   Progressed: {cohort_a_labels['progressed'].sum():,}")
    print(f"   Stagnated: {(~cohort_a_labels['progressed'].astype(bool)).sum():,}")
    
    with torch.no_grad():
        preds_a = get_predictions_for_cohort(model, embeddings, cohort_a_labels, device)
        variables_a = compute_prediction_variables(preds_a)
        
        for var_name, scores in variables_a.items():
            best_metrics = find_best_threshold(
                scores,
                cohort_a_labels['progressed'].values,
                thresholds
            )
            results['cohort_a'][var_name] = best_metrics
    
    # --- Cohort B: Clinical Regression ---
    print("\n🔬 Cohort B: Clinical Regression Detection")
    print(f"   Total pairs: {len(cohort_b_labels):,}")
    print(f"   Regression cases: {cohort_b_labels['is_regression'].sum():,}")
    print(f"   Non-regression: {(~cohort_b_labels['is_regression'].astype(bool)).sum():,}")
    
    with torch.no_grad():
        preds_b = get_predictions_for_cohort(model, embeddings, cohort_b_labels, device)
        variables_b = compute_prediction_variables(preds_b)
        
        for var_name, scores in variables_b.items():
            best_metrics = find_best_threshold(
                scores,
                cohort_b_labels['is_regression'].values,
                thresholds
            )
            results['cohort_b'][var_name] = best_metrics
    
    # Print results
    print_clinical_progression_results(results)
    
    return results


def get_predictions_for_cohort(
    model,
    embeddings: Dict[str, torch.Tensor],
    cohort_labels: pd.DataFrame,
    device
) -> Dict[str, torch.Tensor]:
    """
    Get model predictions for a specific cohort.
    
    Args:
        model: Decoder model
        embeddings: Node embeddings
        cohort_labels: DataFrame with disease_idx, target_idx columns
        device: Torch device
        
    Returns:
        Dict of predictions for each task
    """
    disease_emb = embeddings['disease'][cohort_labels['disease_idx'].values].to(device)
    target_emb = embeddings['target'][cohort_labels['target_idx'].values].to(device)
    
    predictions = model(disease_emb, target_emb)
    
    return predictions


def print_clinical_progression_results(results: Dict):
    """
    Format and print results in GATher Tables 3-4 style.
    
    Args:
        results: Dict with cohort_a and cohort_b results
    """
    print("\n" + "="*80)
    print("📊 CLINICAL PROGRESSION RESULTS (GATher-Style)")
    print("="*80)
    
    cohort_names = {
        'cohort_a': 'COHORT A: POSITIVE PROGRESSION (First-in-Class Detection)',
        'cohort_b': 'COHORT B: CLINICAL REGRESSION (Efficacy Failure Detection)'
    }
    
    for cohort_key, cohort_results in results.items():
        print(f"\n{cohort_names[cohort_key]}")
        print("-" * 110)
        print(f" {'Prediction Variable':<25} {'Thresh.':<8} {'p-value':<12} {'Prec.':<7} {'Rec.':<7} "
              f"{'F1':<7} {'ROC AUC':<8} {'MCC':<7} {'Counts'}")
        print("-" * 110)
        
        for var_name, metrics in cohort_results.items():
            print(f" {var_name:<25} {metrics['threshold']:<8.2f} {metrics['p_val']:<12.2e} "
                  f"{metrics['precision']:<7.3f} {metrics['recall']:<7.3f} {metrics['f1']:<7.3f} "
                  f"{metrics['roc_auc']:<8.3f} {metrics['mcc']:<7.3f} {metrics['counts']}")
    
    print("=" * 110)

