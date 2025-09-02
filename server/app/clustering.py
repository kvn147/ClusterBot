from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional
import numpy as np
import re
import os
from datetime import datetime, timedelta

class PostClusterer:
    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = float(os.getenv('SIMILARITY_THRESHOLD', similarity_threshold))
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        self.post_vectors = {}
        self.active_clusters = {}
        
    def preprocess_text(self, title: str, content: str = "") -> str:
        """Clean and prepare text for similarity comparison"""
        # Combine title and content, prioritizing title
        text = f"{title} {title} {content}"  # Title weighted more heavily
        
        # Remove Reddit-specific formatting
        text = re.sub(r'\[.*?\]', '', text)  # Remove [brackets]
        text = re.sub(r'\(.*?\)', '', text)  # Remove (parentheses)
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove special chars
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
        
        return text.lower()
    
    def extract_domain(self, url: str) -> str:
        """Extract domain from URL for URL-based clustering"""
        if not url:
            return ""
        
        # Simple domain extraction
        if 'reddit.com' in url:
            return ""  # Skip self posts
        
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
            return domain.replace('www.', '')
        except:
            return ""
    
    def find_similar_cluster(self, post: Dict) -> Optional[int]:
        """Find if post belongs to existing cluster"""
        if not self.active_clusters:
            return None
        
        # Extract features from new post
        processed_text = self.preprocess_text(post['title'], post.get('content', ''))
        post_domain = self.extract_domain(post.get('url', ''))
        
        best_cluster_id = None
        best_similarity = 0
        
        # Check against all active clusters
        for cluster_id, cluster_data in self.active_clusters.items():
            # Skip old clusters (older than 24 hours)
            if self._is_cluster_stale(cluster_data):
                continue
            
            # URL domain matching (high priority)
            if post_domain and post_domain == cluster_data.get('domain', ''):
                if post_domain not in ['twitter.com', 'youtube.com']:  # Avoid over-clustering social media
                    return cluster_id
            
            # Text similarity matching
            cluster_text = cluster_data['representative_text']
            
            # Quick keyword overlap check first (faster)
            if self._quick_keyword_overlap(processed_text, cluster_text) < 0.3:
                continue
            
            # Full TF-IDF similarity for promising candidates
            try:
                combined_texts = [cluster_text, processed_text]
                vectors = self.vectorizer.fit_transform(combined_texts)
                similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
                
                if similarity > best_similarity and similarity > self.similarity_threshold:
                    best_similarity = similarity
                    best_cluster_id = cluster_id
            except:
                continue
        
        return best_cluster_id
    
    def _quick_keyword_overlap(self, text1: str, text2: str) -> float:
        """Fast keyword overlap check to filter candidates"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _is_cluster_stale(self, cluster_data: Dict) -> bool:
        """Check if cluster is too old to accept new posts"""
        cluster_age = datetime.utcnow() - cluster_data['created_at']
        return cluster_age > timedelta(hours=24)
    
    def create_cluster(self, post: Dict) -> int:
        """Create new cluster with post as representative"""
        cluster_id = len(self.active_clusters) + 1
        
        processed_text = self.preprocess_text(post['title'], post.get('content', ''))
        domain = self.extract_domain(post.get('url', ''))
        
        self.active_clusters[cluster_id] = {
            'representative_post_id': post['id'],
            'representative_text': processed_text,
            'domain': domain,
            'created_at': datetime.utcnow(),
            'post_count': 1,
            'title': post['title']
        }
        
        return cluster_id
    
    def add_to_cluster(self, cluster_id: int, post: Dict):
        """Add post to existing cluster"""
        if cluster_id in self.active_clusters:
            self.active_clusters[cluster_id]['post_count'] += 1