import os
import pandas as pd
import numpy as np

BASE_DIR = r"C:\Users\aaron\Downloads\music-crs-eda"
DATA_DIR = os.path.join(BASE_DIR, "data")

df_train = pd.read_parquet(os.path.join(DATA_DIR, "train.parquet"))
df_test = pd.read_parquet(os.path.join(DATA_DIR, "test.parquet"))
df_tracks = pd.read_parquet(os.path.join(DATA_DIR, "tracks_metadata.parquet"))
df_users = pd.read_parquet(os.path.join(DATA_DIR, "users_metadata.parquet"))

catalog_track_ids = set(df_tracks['track_id'])
catalog_user_ids = set(df_users['user_id'])

# Check recommended tracks across all sessions
all_recommended_tracks = []
recommended_per_turn = {t: [] for t in range(1, 9)}
success_by_category = {}
success_by_specificity = {}

for df in [df_train, df_test]:
    for idx, row in df.iterrows():
        cat = row['conversation_goal']['category']
        spec = row['conversation_goal']['specificity']
        
        # Assessments
        assess_dict = {a['turn_number']: a['goal_progress_assessment'] for a in row['goal_progress_assessments']}
        
        # Recommendations
        for c in row['conversations']:
            if c.get('role') == 'music':
                t_num = c.get('turn_number')
                tid = c.get('content')
                all_recommended_tracks.append(tid)
                recommended_per_turn[t_num].append(tid)
                
                # evaluate success
                succ = assess_dict.get(t_num)
                if succ:
                    if cat not in success_by_category:
                        success_by_category[cat] = {'moves': 0, 'total': 0}
                    if spec not in success_by_specificity:
                        success_by_specificity[spec] = {'moves': 0, 'total': 0}
                    
                    success_by_category[cat]['total'] += 1
                    success_by_specificity[spec]['total'] += 1
                    if succ == 'MOVES_TOWARD_GOAL':
                        success_by_category[cat]['moves'] += 1
                        success_by_specificity[spec]['moves'] += 1

unique_rec_tracks = set(all_recommended_tracks)
in_catalog = unique_rec_tracks.intersection(catalog_track_ids)

print(f"Total track recommendations made: {len(all_recommended_tracks):,}")
print(f"Unique tracks recommended: {len(unique_rec_tracks):,}")
print(f"Recommended tracks in catalog: {len(in_catalog):,} / {len(unique_rec_tracks):,} ({len(in_catalog)/len(unique_rec_tracks)*100:.2f}%)")
print(f"Catalog tracks ever recommended: {len(in_catalog):,} / {len(catalog_track_ids):,} ({len(in_catalog)/len(catalog_track_ids)*100:.2f}%)")

session_user_ids = set(df_train['user_id']).union(set(df_test['user_id']))
print(f"\nSession users in user metadata catalog: {len(session_user_ids.intersection(catalog_user_ids)):,} / {len(session_user_ids):,} ({len(session_user_ids.intersection(catalog_user_ids))/len(session_user_ids)*100:.2f}%)")

print("\n--- Success Rate by Goal Category ---")
for cat, data in sorted(success_by_category.items()):
    rate = data['moves'] / data['total'] * 100
    print(f"Category {cat}: {rate:.1f}% positive moves ({data['moves']:,}/{data['total']:,})")

print("\n--- Success Rate by Specificity ---")
for spec, data in sorted(success_by_specificity.items()):
    rate = data['moves'] / data['total'] * 100
    print(f"Specificity {spec}: {rate:.1f}% positive moves ({data['moves']:,}/{data['total']:,})")

# Check repeatability of recommendations per session
# Does a session ever repeat the same track?
repeats = 0
for df in [df_train, df_test]:
    for idx, row in df.iterrows():
        tids = [c.get('content') for c in row['conversations'] if c.get('role') == 'music']
        if len(tids) != len(set(tids)):
            repeats += 1
print(f"\nSessions with repeated track recommendations: {repeats:,} / {len(df_train)+len(df_test):,} ({repeats/(len(df_train)+len(df_test))*100:.2f}%)")

