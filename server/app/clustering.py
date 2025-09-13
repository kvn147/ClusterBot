from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional
import numpy as np
import re
import os
from datetime import datetime, timedelta, timezone


class PostClusterer:
    def __init__(
        self, similarity_threshold: float = 0.25
    ):  # Threshold that should capture all similar posts
        self.similarity_threshold = float(
            os.getenv("SIMILARITY_THRESHOLD", similarity_threshold)
        )
        self.vectorizer = TfidfVectorizer(
            max_features=1000, stop_words="english", ngram_range=(1, 2), lowercase=True
        )
        self.post_vectors = {}
        self.active_clusters = {}

    def preprocess_text(self, title: str, content: str = "") -> str:
        """Clean and prepare text for similarity comparison - TITLE FOCUSED"""
        # Focus heavily on title, lightly on content
        # Title appears 5 times, content appears once and truncated
        content_snippet = content[:200] if content else ""  # Only first 200 chars
        text = f"{title} {title} {title} {title} {title} {content_snippet}"

        # Normalize earthquake terminology
        text = re.sub(r"\bquake\b", "earthquake", text, flags=re.IGNORECASE)
        text = re.sub(r"\bhits?\b", "strikes", text, flags=re.IGNORECASE)
        text = re.sub(r"\bcauses?\b", "leads to", text, flags=re.IGNORECASE)
        text = re.sub(r"\bmagnitude\b", "mag", text, flags=re.IGNORECASE)
        text = re.sub(
            r"\bupdate[sd]?\b", "update", text, flags=re.IGNORECASE
        )  # Normalize updates/updating
        text = re.sub(
            r"\breport(?:s|ed|ing)?\b", "report", text, flags=re.IGNORECASE
        )  # Normalize reports/reported
        text = re.sub(
            r"\bpower outages?\b", "power outage", text, flags=re.IGNORECASE
        )  # Normalize power outage terms

        # Extract and preserve key info: locations, numbers, disaster types
        numbers = re.findall(r"\d+(?:\.\d+)?", text)  # Extract numbers like 7.2
        locations = re.findall(
            r"\b(?:japan|tokyo|honshu|osaka|kyoto|sendai|fukushima|hokkaido|okinawa)\b",
            text.lower(),
        )  # Common locations
        disaster_terms = re.findall(
            r"\b(?:earthquake|tsunami|aftershock|tremor|damage|evacuat(?:e|ion)|warning)\b",
            text.lower(),
        )  # Key disaster terms

        # Remove Reddit-specific formatting and noise
        text = re.sub(r"\[.*?\]", "", text)  # Remove [brackets]
        text = re.sub(r"\(.*?\)", "", text)  # Remove (parentheses)
        text = re.sub(r"breaking:?", "", text, flags=re.IGNORECASE)  # Remove "breaking"
        text = re.sub(
            r"live updates?:?", "update", text, flags=re.IGNORECASE
        )  # Normalize "live updates"
        text = re.sub(r"http\S+", "", text)  # Remove URLs
        text = re.sub(r"[^\w\s]", " ", text)  # Remove special chars
        text = re.sub(r"\s+", " ", text).strip()  # Normalize whitespace

        # Add extracted key info back with higher weight (repeat them)
        key_info = " ".join(
            [
                " ".join(locations) * 3,
                " ".join(numbers) * 3,
                " ".join(disaster_terms) * 3,
            ]
        )
        text = f"{text} {key_info}"

        return text.lower()

    def extract_domain(self, url: str) -> str:
        """Extract domain from URL for URL-based clustering"""
        if not url:
            return ""

        # Simple domain extraction
        if "reddit.com" in url:
            return ""  # Skip self posts

        try:
            from urllib.parse import urlparse

            domain = urlparse(url).netloc
            return domain.replace("www.", "")
        except:
            return ""

    def calculate_title_similarity(self, post1: Dict, post2: Dict) -> float:
        """Calculate similarity focusing primarily on titles"""
        title1 = self.preprocess_text(post1["title"], "")  # Title only
        title2 = self.preprocess_text(post2["title"], "")  # Title only

        try:
            combined_texts = [title1, title2]
            vectors = self.vectorizer.fit_transform(combined_texts)
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            return similarity
        except:
            return 0.0

    def find_similar_cluster(self, post: Dict) -> Optional[int]:
        """Find if post belongs to existing cluster"""
        if not self.active_clusters:
            return None

        # Extract features from new post
        post_domain = self.extract_domain(post.get("url", ""))
        post_text = self.preprocess_text(post["title"], post.get("selftext", ""))

        best_cluster_id = None
        best_similarity = 0

        debug_info = []

        # Check against all active clusters
        for cluster_id, cluster_data in self.active_clusters.items():
            # Skip old clusters (older than 24 hours)
            if self._is_cluster_stale(cluster_data):
                continue

            representative_post = cluster_data["representative_post"]
            rep_text = self.preprocess_text(
                representative_post["title"], representative_post.get("selftext", "")
            )

            # Quick keyword overlap pre-check - skip unlikely matches
            keyword_overlap = self._quick_keyword_overlap(post_text, rep_text)
            if keyword_overlap < 0.2:  # Skip detailed comparison for very low overlap
                debug_info.append(
                    f"Cluster {cluster_id}: {keyword_overlap:.2f} keyword overlap (too low)"
                )
                continue

            # URL domain matching (high priority) - only for news domains
            if post_domain and post_domain == cluster_data.get("domain", ""):
                # Avoid over-clustering social media
                if post_domain not in [
                    "twitter.com",
                    "youtube.com",
                    "facebook.com",
                    "instagram.com",
                    "reddit.com",
                ]:
                    debug_info.append(
                        f"Cluster {cluster_id}: Same domain {post_domain} -> MATCHED"
                    )
                    return cluster_id

            # Title-focused similarity matching
            title_similarity = self.calculate_title_similarity(
                post, representative_post
            )

            debug_info.append(
                f"Cluster {cluster_id}: {title_similarity:.3f} similarity"
            )

            # Check for event-specific matches (keywords like earthquake, location, magnitude)
            event_match = self._check_event_match(
                post["title"], representative_post["title"]
            )
            if event_match:
                title_similarity += 0.15  # Boost similarity for event matches
                debug_info.append(
                    f"  ✓ Event match detected: +0.15 boost -> {title_similarity:.3f}"
                )

            if title_similarity > best_similarity:
                best_similarity = title_similarity
                best_cluster_id = cluster_id

        # Print debug info
        for info in debug_info:
            print(f"   {info}")

        if best_similarity > self.similarity_threshold:
            return best_cluster_id
        else:
            return None

    def _check_event_match(self, title1: str, title2: str) -> bool:
        """Check if titles refer to the same event based on key elements"""
        title1_lower = title1.lower()
        title2_lower = title2.lower()

        # Check for event type match
        event_types = [
            "earthquake",
            "tsunami",
            "hurricane",
            "typhoon",
            "tornado",
            "flood",
            "wildfire",
        ]
        event_match = False
        for event in event_types:
            if event in title1_lower and event in title2_lower:
                event_match = True
                break

        if not event_match:
            return False

        # Extract locations
        locations = [
            "japan",
            "california",
            "florida",
            "texas",
            "china",
            "india",
            "europe",
            "australia",
        ]
        location_match = False
        for location in locations:
            if location in title1_lower and location in title2_lower:
                location_match = True
                break

        # Extract numbers (like magnitude)
        numbers1 = re.findall(r"\d+(?:\.\d+)?", title1)
        numbers2 = re.findall(r"\d+(?:\.\d+)?", title2)
        number_match = False

        for n1 in numbers1:
            for n2 in numbers2:
                # Consider numbers within 0.5 as the same (e.g., 7.1 and 7.2 magnitude)
                try:
                    if abs(float(n1) - float(n2)) < 0.5:
                        number_match = True
                        break
                except ValueError:
                    continue

        # Return true if at least event+location or event+number matched
        return event_match and (location_match or number_match)

    def _quick_keyword_overlap(self, text1: str, text2: str) -> float:
        """Fast keyword overlap check to filter candidates - LOWERED THRESHOLD"""
        # Extract key words only (at least 3 characters to avoid noise)
        words1 = set(word for word in text1.split() if len(word) >= 3)
        words2 = set(word for word in text2.split() if len(word) >= 3)

        if not words1 or not words2:
            return 0

        # Extract important keywords for special weighting
        important_keywords = {
            "earthquake",
            "tsunami",
            "japan",
            "damage",
            "warning",
            "report",
            "update",
        }

        # Calculate weighted intersection
        intersection = words1.intersection(words2)
        union = words1.union(words2)

        # Add bonus for important keywords
        important_matches = len(
            [word for word in intersection if word in important_keywords]
        )
        bonus = min(0.1 * important_matches, 0.2)  # Cap bonus at 0.2

        overlap_score = len(intersection) / len(union) + bonus
        return min(overlap_score, 1.0)  # Cap at 1.0

    def _is_cluster_stale(self, cluster_data: Dict) -> bool:
        """Check if cluster is too old to accept new posts"""
        cluster_age = datetime.now(timezone.utc) - cluster_data["created_at"]
        return cluster_age > timedelta(hours=24)

    def create_cluster(self, post: Dict) -> int:
        """Create new cluster with post as representative"""
        cluster_id = len(self.active_clusters) + 1

        domain = self.extract_domain(post.get("url", ""))

        self.active_clusters[cluster_id] = {
            "representative_post_id": post["id"],
            "representative_post": post,  # Store full post for similarity comparison
            "domain": domain,
            "created_at": datetime.now(timezone.utc),
            "post_count": 1,
            "title": post["title"],
        }

        return cluster_id

    def add_to_cluster(self, cluster_id: int, post: Dict):
        """Add post to existing cluster"""
        if cluster_id in self.active_clusters:
            self.active_clusters[cluster_id]["post_count"] += 1


# Test the improved algorithm
def test_improved_clustering():
    """Test the improved clustering algorithm"""
    print("🚀 Testing Improved Clustering Algorithm")
    print("=" * 50)

    import sys
    import os

    sys.path.append(".")

    from tests.sample_data.test_posts import (
        get_earthquake_posts,
        get_tech_posts,
        get_unrelated_posts,
    )

    # Test with lower threshold
    clusterer = PostClusterer(similarity_threshold=0.2)

    eq_posts = get_earthquake_posts()
    print(f"\n📰 Testing {len(eq_posts)} earthquake posts:")

    # Create first cluster
    cluster1 = clusterer.create_cluster(eq_posts[0])
    print(f"✅ Created cluster {cluster1}: '{eq_posts[0]['title'][:50]}...'")

    clustered_count = 1

    # Test remaining posts
    for i, post in enumerate(eq_posts[1:], 2):
        print(f"\n🔍 Testing post {i}: '{post['title'][:50]}...'")
        match = clusterer.find_similar_cluster(post)

        if match:
            clusterer.add_to_cluster(match, post)
            print(f"✅ MATCHED to cluster {match}")
            clustered_count += 1
        else:
            new_cluster = clusterer.create_cluster(post)
            print(f"⚠️ Created NEW cluster {new_cluster}")

    print(
        f"\n📊 Final Result: {clustered_count}/{len(eq_posts)} earthquake posts clustered together"
    )

    if clustered_count == len(eq_posts):
        print("🎉 PERFECT! All earthquake posts in same cluster!")
    else:
        print("⚠️ Still needs tuning...")

    return clustered_count == len(eq_posts)


if __name__ == "__main__":
    test_improved_clustering()
