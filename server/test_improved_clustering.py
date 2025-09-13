#!/usr/bin/env python3

from app.clustering import PostClusterer
from tests.sample_data.test_posts import get_earthquake_posts, get_tech_posts


def test_earthquake_clustering():
    """Test if earthquake posts are correctly clustered together"""
    print("\n🌎 Testing Earthquake Post Clustering")
    print("=" * 60)

    # Create clusterer with our improved algorithm
    clusterer = PostClusterer(similarity_threshold=0.25)

    # Get test earthquake posts
    earthquake_posts = get_earthquake_posts()

    print(f"Found {len(earthquake_posts)} earthquake posts for testing")

    # Create first cluster with first post
    cluster1 = clusterer.create_cluster(earthquake_posts[0])
    print(f"\nCreated first cluster with post: '{earthquake_posts[0]['title']}'")

    clustered_count = 1
    cluster_assignments = {1: [earthquake_posts[0]["title"]]}

    # Try to cluster remaining posts
    for i, post in enumerate(earthquake_posts[1:], 2):
        print(f"\n🔍 Testing post {i}: '{post['title']}'")

        match = clusterer.find_similar_cluster(post)

        if match:
            clusterer.add_to_cluster(match, post)
            print(f"✅ MATCHED to cluster {match}")
            cluster_assignments.setdefault(match, []).append(post["title"])
            clustered_count += 1
        else:
            new_cluster = clusterer.create_cluster(post)
            print(f"❌ Created NEW cluster {new_cluster}")
            cluster_assignments[new_cluster] = [post["title"]]

    print("\n\n📊 Clustering Results:")
    print("=" * 60)

    for cluster_id, titles in cluster_assignments.items():
        print(f"\n📌 Cluster {cluster_id} ({len(titles)} posts):")
        for i, title in enumerate(titles, 1):
            print(f"  {i}. {title}")

    print("\n📊 Summary:")
    print(f"  - Total posts: {len(earthquake_posts)}")
    print(f"  - Number of clusters: {len(cluster_assignments)}")

    if len(cluster_assignments) == 1:
        print("🎉 PERFECT! All earthquake posts were grouped into a single cluster!")
    else:
        print(
            f"⚠️ Not ideal. Posts were split into {len(cluster_assignments)} clusters."
        )

    return cluster_assignments


def test_different_thresholds():
    """Test how different similarity thresholds affect clustering"""
    print("\n🔍 Testing Different Similarity Thresholds")
    print("=" * 60)

    thresholds = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5]
    results = {}

    earthquake_posts = get_earthquake_posts()

    for threshold in thresholds:
        print(f"\n⚙️ Testing with threshold: {threshold}")
        clusterer = PostClusterer(similarity_threshold=threshold)

        # Create first cluster
        cluster1 = clusterer.create_cluster(earthquake_posts[0])

        cluster_count = 1
        post_assignments = {1: 1}  # Cluster ID: Count of posts

        # Try to cluster remaining posts
        for post in earthquake_posts[1:]:
            match = clusterer.find_similar_cluster(post)

            if match:
                clusterer.add_to_cluster(match, post)
                post_assignments[match] = post_assignments.get(match, 0) + 1
            else:
                new_cluster = clusterer.create_cluster(post)
                cluster_count += 1
                post_assignments[new_cluster] = 1

        status = "✅ IDEAL" if cluster_count == 1 else "❌ FRAGMENTED"
        print(f"Result: {cluster_count} clusters {status}")

        results[threshold] = {
            "cluster_count": cluster_count,
            "distribution": post_assignments,
        }

    print("\n📊 Threshold Analysis Summary:")
    print("=" * 40)

    best_threshold = None
    best_cluster_count = float("inf")

    for threshold, data in results.items():
        cluster_count = data["cluster_count"]
        distribution = ", ".join(
            [f"Cluster {c}: {n} posts" for c, n in data["distribution"].items()]
        )

        print(f"Threshold {threshold}: {cluster_count} clusters ({distribution})")

        if cluster_count < best_cluster_count:
            best_cluster_count = cluster_count
            best_threshold = threshold

    print(
        f"\n🏆 Best threshold: {best_threshold} (resulted in {best_cluster_count} clusters)"
    )

    if best_cluster_count == 1:
        print("✨ Perfect clustering achieved!")
    else:
        print(
            f"📝 Consider further tuning the algorithm to achieve single-cluster grouping."
        )


if __name__ == "__main__":
    print("🚀 ClusterBot Improved Clustering Test")
    print("=" * 60)

    earthquake_clusters = test_earthquake_clustering()
    test_different_thresholds()
