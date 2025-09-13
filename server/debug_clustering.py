#!/usr/bin/env python3
import sys
import os

sys.path.append(".")

from app.clustering import PostClusterer
from tests.sample_data.test_posts import get_earthquake_posts, get_tech_posts
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def debug_text_preprocessing():
    """Debug how text preprocessing works"""
    print("🔧 Testing Text Preprocessing")
    print("=" * 50)

    clusterer = PostClusterer()
    eq_posts = get_earthquake_posts()

    print("Original titles:")
    for i, post in enumerate(eq_posts, 1):
        print(f"{i}. {post['title']}")

    print("\nProcessed text:")
    processed_texts = []
    for i, post in enumerate(eq_posts, 1):
        processed = clusterer.preprocess_text(post["title"], post.get("selftext", ""))
        processed_texts.append(processed)
        print(f"{i}. {processed}")

    return processed_texts


def debug_similarity_calculation():
    """Debug similarity scores between earthquake posts"""
    print("\n📊 Testing Similarity Calculations")
    print("=" * 50)

    clusterer = PostClusterer()
    eq_posts = get_earthquake_posts()

    # Get processed texts
    processed_texts = []
    for post in eq_posts:
        processed = clusterer.preprocess_text(post["title"], post.get("selftext", ""))
        processed_texts.append(processed)

    print("Similarity matrix (should be high for earthquake posts):")
    print("Post pairs and their similarity scores:")
    print()

    # Calculate pairwise similarities
    vectorizer = TfidfVectorizer(
        max_features=1000, stop_words="english", ngram_range=(1, 2)
    )

    try:
        vectors = vectorizer.fit_transform(processed_texts)
        similarity_matrix = cosine_similarity(vectors)

        for i in range(len(eq_posts)):
            for j in range(i + 1, len(eq_posts)):
                similarity = similarity_matrix[i][j]
                status = (
                    "✅ MATCH"
                    if similarity > 0.8
                    else "⚠️ THRESHOLD" if similarity > 0.6 else "❌ NO MATCH"
                )
                print(f"Post {i+1} vs Post {j+1}: {similarity:.3f} {status}")
                print(f"  '{eq_posts[i]['title'][:40]}...'")
                print(f"  '{eq_posts[j]['title'][:40]}...'")
                print()

        # Find the highest similarity
        max_similarity = 0
        for i in range(len(eq_posts)):
            for j in range(i + 1, len(eq_posts)):
                if similarity_matrix[i][j] > max_similarity:
                    max_similarity = similarity_matrix[i][j]

        print(f"📈 Highest similarity between earthquake posts: {max_similarity:.3f}")
        print(f"🎯 Current threshold: 0.8")
        print(f"💡 Recommended threshold: {max_similarity - 0.05:.2f}")

        return max_similarity

    except Exception as e:
        print(f"❌ Error calculating similarities: {e}")
        return None


def test_different_thresholds():
    """Test clustering with different similarity thresholds"""
    print("\n🎯 Testing Different Similarity Thresholds")
    print("=" * 50)

    eq_posts = get_earthquake_posts()
    thresholds = [0.5, 0.6, 0.7, 0.8]

    for threshold in thresholds:
        print(f"\n🔬 Testing threshold: {threshold}")
        clusterer = PostClusterer(similarity_threshold=threshold)

        # Create cluster with first post
        cluster1 = clusterer.create_cluster(eq_posts[0])
        clustered_count = 1

        # Test remaining posts
        for post in eq_posts[1:]:
            match = clusterer.find_similar_cluster(post)
            if match:
                clusterer.add_to_cluster(match, post)
                clustered_count += 1
                print(f"   ✅ Post matched to cluster {match}")
            else:
                new_cluster = clusterer.create_cluster(post)
                print(f"   ❌ Post created new cluster {new_cluster}")

        result = f"{clustered_count}/{len(eq_posts)} posts in main cluster"
        score = (
            "🎯 PERFECT!"
            if clustered_count == len(eq_posts)
            else "⚠️ Split" if clustered_count > 1 else "❌ All separate"
        )
        print(f"   📊 Result: {result} {score}")


def test_keyword_overlap():
    """Test the quick keyword overlap function"""
    print("\n🔍 Testing Keyword Overlap Detection")
    print("=" * 50)

    clusterer = PostClusterer()
    eq_posts = get_earthquake_posts()

    post1_processed = clusterer.preprocess_text(eq_posts[0]["title"])

    print(f"Reference post: '{eq_posts[0]['title'][:50]}...'")
    print(f"Processed: '{post1_processed}'")
    print()

    for i, post in enumerate(eq_posts[1:], 1):
        post_processed = clusterer.preprocess_text(post["title"])
        overlap = clusterer._quick_keyword_overlap(post1_processed, post_processed)

        status = "✅ PASS" if overlap >= 0.3 else "❌ FILTERED OUT"
        print(f"Post {i+1}: {overlap:.3f} overlap {status}")
        print(f"  '{post['title'][:50]}...'")
        print(f"  Processed: '{post_processed}'")
        print()


def main():
    """Run all debug tests"""
    print("🚨 ClusterBot Debug Suite")
    print("=" * 60)
    print("Your earthquake posts are splitting into separate clusters.")
    print("Let's find out why...")
    print()

    # Test 1: Text preprocessing
    processed_texts = debug_text_preprocessing()

    # Test 2: Similarity calculation
    max_similarity = debug_similarity_calculation()

    # Test 3: Different thresholds
    test_different_thresholds()

    # Test 4: Keyword overlap
    test_keyword_overlap()

    # Recommendations
    print("\n💡 Recommendations")
    print("=" * 30)

    if max_similarity and max_similarity < 0.8:
        print(
            f"🎯 Lower your similarity threshold from 0.8 to {max_similarity - 0.05:.2f}"
        )
        print(f"   Your earthquake posts have max similarity of {max_similarity:.3f}")

    print("🔧 You can also try:")
    print("   1. Improve text preprocessing (remove more noise)")
    print("   2. Weight title text more heavily")
    print("   3. Add URL domain matching")
    print("   4. Use sentence transformers instead of TF-IDF")

    print(f"\n🚀 Quick Fix: Update your clustering.py:")
    print(f"   PostClusterer(similarity_threshold=0.6)  # Instead of 0.8")


if __name__ == "__main__":
    main()
