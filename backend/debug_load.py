import torch
import os
p = os.path.join(os.path.dirname(__file__), '..', 'models','best_chest_model_8320.pth')
print('Checkpoint path:', p)
ck = torch.load(p, map_location='cpu')
print('Type:', type(ck))
try:
    keys = list(ck.keys())
    print('Checkpoint keys sample:', keys[:10])
except Exception as e:
    print('Not a dict, error listing keys:', e)

st = ck.get('model_state_dict', ck) if isinstance(ck, dict) else ck
print('State dict length:', len(st))
# Check some likely classifier keys and shapes
for k in ['classifier.1.weight','classifier.4.weight','classifier.weight','model.classifier.4.weight','model.classifier.1.weight','classifier.4.bias']:
    if k in st:
        try:
            print(k, '->', st[k].shape)
        except Exception as e:
            print(k, 'found but cannot show shape:', e)

print('\nFirst 30 state keys:')
for i,k in enumerate(list(st.keys())[:30]):
    print(i+1, k)
