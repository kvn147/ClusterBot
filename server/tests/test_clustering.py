#!/usr/bin/env python3
"""Quick test to validate clustering with sample data"""

import sys
import os

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.clustering import PostClusterer
from tests.sample_data.test_posts import (
    get_earthquake_posts, 
    get_tech_posts, 
    get_unrelated_posts,
    get_duplicate_url_posts
)

def test_clustering_with_sample_data():
    """Test clustering algorithm with our sample posts"""
    print("🧪 Testing ClusterBot with Sample Data")
    print("=" * 50)
    
    # Initialize clusterer with slightly lower threshold for testing
    clusterer = PostClusterer(similarity_threshold=0.6)
    
    # Test 1: Earthquake posts should cluster together
    print("\n📰 Test 1: Earthquake Posts Should Cluster Together")
    earthquake_posts = get_earthquake_posts()
    
    # Create cluster with first earthquake post
    first_post = earthquake_posts[0]
    cluster1_id = clusterer.create_cluster(first_post)
    print(f"✅ Created cluster {cluster1_id} for: '{first_post['title'][:50]}...'")
    
    # Test remaining earthquake posts
    earthquake_clusters = {}
    for i, post in enumerate(earthquake_posts[1:], 1):
        cluster_id = clusterer.find_similar_cluster(post)
        if cluster_id:
            clusterer.add_to_cluster(cluster_id, post)
            print(f"✅ Post {i+1} matched to cluster {cluster_id}: '{post['title'][:50]}...'")
            earthquake_clusters[cluster_id] = earthquake_clusters.get(cluster_id, 0) + 1
        else:
            new_cluster = clusterer.create_cluster(post)
            print(f"⚠️  Post {i+1} created NEW cluster {new_cluster}: '{post['title'][:50]}...'")
            earthquake_clusters[new_cluster] = 1
    
    print(f"📊 Earthquake clustering result: {len(earthquake_clusters)} cluster(s)")
    
    # Test 2: Tech posts should cluster together (separately from earthquake)
    print("\n💻 Test 2: Tech Posts Should Cluster Together")
    tech_posts = get_tech_posts()
    
    tech_clusters = {}
    for i, post in enumerate(tech_posts):
        cluster_id = clusterer.find_similar_cluster(post)
        if cluster_id:
            clusterer.add_to_cluster(cluster_id, post)
            print(f"✅ Tech post {i+1} matched to cluster {cluster_id}: '{post['title'][:50]}...'")
            tech_clusters[cluster_id] = tech_clusters.get(cluster_id, 0) + 1
        else:
            new_cluster = clusterer.create_cluster(post)
            print(f"🆕 Tech post {i+1} created cluster {new_cluster}: '{post['title'][:50]}...'")
            tech_clusters[new_cluster] = 1
    
    print(f"📊 Tech clustering result: {len(tech_clusters)} cluster(s)")
    
    # Test 3: Unrelated posts should NOT cluster with anything
    print("\n🔀 Test 3: Unrelated Posts Should NOT Cluster")
    unrelated_posts = get_unrelated_posts()
    
    unrelated_matches = 0
    for i, post in enumerate(unrelated_posts):
        cluster_id = clusterer.find_similar_cluster(post)
        if cluster_id:
            print(f"⚠️  Unrelated post {i+1} matched to existing cluster {cluster_id}: '{post['title'][:50]}...'")
            unrelated_matches += 1
        else:
            new_cluster = clusterer.create_cluster(post)
            print(f"✅ Unrelated post {i+1} correctly created new cluster {new_cluster}: '{post['title'][:50]}...'")
    
    print(f"📊 Unrelated posts incorrectly clustered: {unrelated_matches}/{len(unrelated_posts)}")
    
    # Test 4: Posts with same URL should definitely cluster together
    print("\n🔗 Test 4: Same URL Posts Should Cluster Together")
    duplicate_posts = get_duplicate_url_posts()
    
    if duplicate_posts:
        first_dup = duplicate_posts[0]
        dup_cluster_id = clusterer.create_cluster(first_dup)
        print(f"🆕 Created cluster {dup_cluster_id} for: '{first_dup['title'][:50]}...'")
        
        for i, post in enumerate(duplicate_posts[1:], 1):
            cluster_id = clusterer.find_similar_cluster(post)
            if cluster_id == dup_cluster_id:
                print(f"✅ Duplicate URL post {i+1} correctly matched: '{post['title'][:50]}...'")
            elif cluster_id:
                print(f"⚠️  Duplicate URL post {i+1} matched wrong cluster {cluster_id}: '{post['title'][:50]}...'")
            else:
                print(f"❌ Duplicate URL post {i+1} failed to cluster: '{post['title'][:50]}...'")
    
    # Summary
    print("\n📈 Clustering Performance Summary")
    print("=" * 50)
    print(f"Total active clusters: {len(clusterer.active_clusters)}")
    print(f"Similarity threshold: {clusterer.similarity_threshold}")
    
    # Show cluster distribution
    for cluster_id, cluster_data in clusterer.active_clusters.items():
        print(f"Cluster {cluster_id}: {cluster_data['post_count']} posts - '{cluster_data['title'][:40]}...'")

def test_preprocessing():
    """Test text preprocessing"""
    print("\n🔧 Testing Text Preprocessing")
    print("=" * 30)
    
    clusterer = PostClusterer()
    
    test_cases = [
        ("Breaking: Major [UPDATE] earthquake hits Japan (7.2 magnitude)", "earthquake japan"),
        ("Tesla announces new Model Y http://example.com", "tesla announces new model"),
        ("LIVE: Election results!!! #breaking", "live election results breaking")
    ]
    
    for original, expected_contains in test_cases:
        processed = clusterer.preprocess_text(original)
        contains_expected = all(word in processed for word in expected_contains.split())
        status = "✅" if contains_expected else "❌"
        print(f"{status} '{original}' -> '{processed}'")

if __name__ == "__main__":
    try:
        test_preprocessing()
        test_clustering_with_sample_data()
        print("\n🎉 All tests completed! Check the results above.")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running from the server/ directory")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()