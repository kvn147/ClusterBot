import sys
import os

sys.path.append(".")

from app.clustering import PostClusterer
from tests.sample_data.test_posts import (
    get_earthquake_posts,
    get_tech_posts,
    get_unrelated_posts,
)

clusterer = PostClusterer(similarity_threshold=0.7)

print("Testing earthquake posts...")
eq_posts = get_earthquake_posts()
cluster1 = clusterer.create_cluster(eq_posts[0])
print(f"Created cluster {cluster1}")

for post in eq_posts[1:]:
    match = clusterer.find_similar_cluster(post)
    if match:
        print(f"✅ Matched to cluster {match}: {post['title'][:50]}...")
        clusterer.add_to_cluster(match, post)
    else:
        new_cluster = clusterer.create_cluster(post)
        print(f"⚠️ Created NEW cluster {new_cluster}: {post['title'][:50]}...")

print(f"\nTotal clusters: {len(clusterer.active_clusters)}")
