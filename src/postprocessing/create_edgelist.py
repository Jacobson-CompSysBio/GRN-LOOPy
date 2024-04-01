import pandas as pd

def parse_importances(importance_data: pd.DataFrame, weigh_by_acc:bool) -> pd.DataFrame:
    features = importance_data['features'].split('|')
    acc = float(importance_data['r2'])
    feature_importances = importance_data['feature_imps'].split('|')
    feature_name = importance_data['feature']
    df = pd.DataFrame({
        'from': features,
        'to': feature_name,
        'weight': feature_importances
    })
    df['weight'] = df['weight'].apply(float)
    df['weight'] =     df['weight'] /  df['weight'].sum()
    if weigh_by_acc: 
        df['weight']  = df['weight']  * acc
    return df[ df['weight'] > 0 ]

def create_edgelist(df: pd.DataFrame, weigh_by_acc: bool):
    output_values = df.apply(lambda x: parse_importances(x, weigh_by_acc), axis=1)
    output_edgelist = pd.concat(output_values.values)
    return output_edgelist.reset_index(drop=True) # .sort_values(by='weight', ascending=False).

