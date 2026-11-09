import pandas as pd
import json

df = pd.read_parquet(r"C:\Users\aaron\Downloads\music-crs-eda\data\train.parquet")
sample = df.iloc[0]

print("=== Sample Record Keys ===")
print(df.columns.tolist())

print("\n=== Conversation Goal Sample ===")
print(json.dumps(sample['conversation_goal'], indent=2))

print(f"\n=== Conversations Count: {len(sample['conversations'])} ===")
roles = [c.get('role') for c in sample['conversations']]
turn_numbers = [c.get('turn_number') for c in sample['conversations']]
print("Roles:", roles)
print("Turn numbers:", turn_numbers)

print("\n=== First 6 Conversation Turns ===")
for i, c in enumerate(sample['conversations'][:6]):
    print(f"Turn {c.get('turn_number')} [{c.get('role')}]:")
    print(f"  Content: {c.get('content')}")
    if c.get('thought'):
        print(f"  Thought: {c.get('thought')}")

print("\n=== Goal Progress Assessments ===")
print(json.dumps(list(sample['goal_progress_assessments']), indent=2))

# Check goal categories description/examples
print("\n=== Examples of Categories A through K ===")
cat_samples = {}
for idx, row in df.iterrows():
    cg = row['conversation_goal']
    cat = cg.get('category')
    if cat not in cat_samples:
        cat_samples[cat] = {
            'category': cat,
            'specificity': cg.get('specificity'),
            'listener_goal': cg.get('listener_goal')
        }
    if len(cat_samples) == 11:
        break

for cat in sorted(cat_samples.keys()):
    print(f"Category {cat} (Spec: {cat_samples[cat]['specificity']}): {cat_samples[cat]['listener_goal']}")

