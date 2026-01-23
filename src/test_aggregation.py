import pandas as pd
import numpy as np

def harmonic_sum(scores):
    # Standard Harmonic Sum: sum(s / (i+1)^2) ? No, standard is usually different.
    # Open Targets Harmonic Sum: score_1 + score_2 / (2^2) + score_3 / (3^2)...
    # Sorted descending.
    scores = sorted(scores, reverse=True)
    return sum(s / ((i + 1) ** 2) for i, s in enumerate(scores))

def aggregate_scores(scores, method='harmonic_sum'):
    if method == 'max':
        return max(scores)
    elif method == 'harmonic_sum':
        return harmonic_sum(scores)
    else:
        return max(scores)

def test_aggregation_logic():
    print("🧪 Testing Aggregation Logic...")
    
    # Mock Data:
    # 1. Clinical Trial: Scores [0.5, 0.8, 0.2] -> Expect MAX = 0.8
    # 2. Other Type: Scores [0.5, 0.5] -> Expect Harmonic Sum ~ 0.5 + 0.5/4 = 0.625
    
    data = {
        'sourceId': ['d1']*3 + ['d2']*2,
        'targetId': ['t1']*3 + ['t2']*2,
        'relation': [
            'clinical_trial::chembl', 'clinical_trial::chembl', 'clinical_trial::chembl',
            'genetic_association::eva', 'genetic_association::eva'
        ],
        'score': [0.5, 0.8, 0.2, 0.5, 0.5],
        # Other required cols for groupby
        'source_type': ['disease']*3 + ['disease']*2,
        'target_type': ['target']*3 + ['target']*2,
        'datasourceId': ['chembl']*3 + ['eva']*2,
    }
    
    cumulative_edges = pd.DataFrame(data)
    group_cols = ['sourceId', 'targetId', 'source_type', 'target_type', 'relation', 'datasourceId']
    aggregation_method = 'harmonic_sum' # Default config
    
    print("\nInput Data:")
    print(cumulative_edges[['sourceId', 'relation', 'score']])
    
    # --- LOGIC UNDER TEST (Copied from build_event_list.py) ---
    
    # Split into Clinical (MAX) and Others (Harmonic Sum)
    mask_clinical = cumulative_edges['relation'].str.contains('clinical_trial', case=False)
    clinical_edges = cumulative_edges[mask_clinical]
    other_edges = cumulative_edges[~mask_clinical]
    
    dfs_to_concat = []
    
    # 1. Clinical Trials -> MAX
    if not clinical_edges.empty:
        clinical_agg = clinical_edges.groupby(group_cols, as_index=False)['score'].max()
        dfs_to_concat.append(clinical_agg)
        
    # 2. Others -> Harmonic Sum
    if not other_edges.empty:
        other_agg = other_edges.groupby(group_cols, as_index=False).agg({
            'score': lambda x: aggregate_scores(x.values, method=aggregation_method)
        })
        dfs_to_concat.append(other_agg)
        
    year_scores = pd.concat(dfs_to_concat, ignore_index=True)
    
    # --- VERIFICATION ---
    print("\nAggregated Results:")
    print(year_scores[['sourceId', 'relation', 'score']])
    
    # Check Clinical (d1)
    res_d1 = year_scores[year_scores['sourceId'] == 'd1']['score'].iloc[0]
    expected_d1 = 0.8
    assert np.isclose(res_d1, expected_d1), f"Clinical Error: Expected {expected_d1}, got {res_d1}"
    print(f"✅ Clinical Trial Aggregation: {res_d1} (Expected Max)")
    
    # Check Other (d2)
    res_d2 = year_scores[year_scores['sourceId'] == 'd2']['score'].iloc[0]
    expected_d2 = 0.5 + (0.5 / 4) # 0.625
    assert np.isclose(res_d2, expected_d2), f"Other Error: Expected {expected_d2}, got {res_d2}"
    print(f"✅ Other Logic Aggregation: {res_d2} (Expected Harmonic Sum)")
    
    print("\n🎉 Test Passed!")

if __name__ == "__main__":
    test_aggregation_logic()
