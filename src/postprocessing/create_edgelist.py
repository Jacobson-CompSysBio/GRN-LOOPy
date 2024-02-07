import pandas as pd

def parse_importances(importance_data):
    features = importance_data['features'].split('|')
    feature_importances = importance_data['feature_imps'].split('|')
    feature_name = importance_data['feature']
    df = pd.DataFrame({
        'from': features,
        'to': feature_name,
        'weight': feature_importances
    })
    df['weight'] = df['weight'].apply(float)
    df['weight'] =     df['weight'] /  df['weight'].sum()
    return df[ df['weight'] > 0 ]

def create_edgelist(df: pd.DataFrame):
    output_values = df.apply(parse_importances, axis=1)
    output_edgelist = pd.concat(output_values.values)
    return output_edgelist.reset_index(drop=True) # .sort_values(by='weight', ascending=False).

