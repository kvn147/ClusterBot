import json
import os
from typing import Dict, List

def load_sample_posts() -> Dict[str, List[Dict]]:
    """Load sample posts from JSON file"""
    current_dir = os.path.dirname(__file__)
    json_path = os.path.join(current_dir, 'sample_posts.json')
    
    with open(json_path, 'r') as f:
        return json.load(f)

def get_earthquake_posts() -> List[Dict]:
    """Get earthquake-related posts for clustering tests"""
    return load_sample_posts()["earthquake_cluster"]

def get_tech_posts() -> List[Dict]:
    """Get tech-related posts for clustering tests"""
    return load_sample_posts()["tech_cluster"]

def get_climate_posts() -> List[Dict]:
    """Get climate-related posts for clustering tests"""
    return load_sample_posts()["climate_cluster"]

def get_unrelated_posts() -> List[Dict]:
    """Get unrelated posts that shouldn't cluster together"""
    return load_sample_posts()["unrelated"]

def get_duplicate_url_posts() -> List[Dict]:
    """Get posts with same URL but different titles (should cluster together)"""
    return load_sample_posts()["duplicate_urls"]

def get_all_posts() -> List[Dict]:
    """Get all sample posts as a flat list"""
    data = load_sample_posts()
    all_posts = []
    for category in data.values():
        all_posts.extend(category)
    return all_posts

def get_posts_for_clustering_test() -> Dict[str, List[Dict]]:
    """Get posts organized by expected clusters for testing"""
    return {
        "earthquake_cluster": get_earthquake_posts(),
        "tech_cluster": get_tech_posts(), 
        "climate_cluster": get_climate_posts(),
        "should_not_cluster": get_unrelated_posts()
    }

def print_sample_data_summary():
    """Print summary of available test data"""
    data = load_sample_posts()
    print("📊 Sample Data Summary:")
    print(f"━━━━━━━━━━━━━━━━━━━━━━")
    
    for category, posts in data.items():
        print(f"📁 {category}: {len(posts)} posts")
        for post in posts:
            print(f"   • {post['title'][:60]}...")
        print()
    
    print(f"📈 Total posts: {len(get_all_posts())}")

if __name__ == "__main__":
    # Test the data loading
    print_sample_data_summary()
    
    # Test individual functions
    print("\n🧪 Testing data access functions:")
    print(f"Earthquake posts: {len(get_earthquake_posts())}")
    print(f"Tech posts: {len(get_tech_posts())}")
    print(f"Unrelated posts: {len(get_unrelated_posts())}")
    
    # Show first earthquake post as example
    eq_posts = get_earthquake_posts()
    if eq_posts:
        first_post = eq_posts[0]
        print(f"\n📰 Sample post structure:")
        for key, value in first_post.items():
            if key == 'selftext' and len(str(value)) > 50:
                print(f"   {key}: {str(value)[:50]}...")
            else:
                print(f"   {key}: {value}")