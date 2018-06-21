import pickle
import argparse
import pandas as pd
from quantum_gravity_callback import QuantumGravityCallback

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, default=200, help='submission file', required=True)
args = parser.parse_args()
# yapf: enable

original = pd.read_csv(args.input)
df_test = pd.read_feather('df_test')
df_test = df_test[['item_id', 'category_name']].copy()
original['category_name'] = df_test['category_name']

gravity = QuantumGravityCallback()

original['deal_probability'] = original.apply(
    lambda row: gravity.apply(row), axis=1)

subm = original[['item_id', 'deal_probability']].copy()
subm.to_csv(args.input + '.gravity', index=False)
