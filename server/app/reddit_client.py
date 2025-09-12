import praw
import os
from dotenv import load_dotenv
from typing import List, Dict
import time

load_dotenv()

class RedditClient:
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent="ClusterBot/1.0 by /u/yourusername"  # Better user agent
        )

    def get_new_posts(self, subreddit: str, limit: int = 10) -> List[Dict]:
        """Fetch new posts from a subreddit - matches your test file"""
        posts = []
        try:
            subreddit_instance = self.reddit.subreddit(subreddit)
            for submission in subreddit_instance.new(limit=limit):  # Changed to .new()
                posts.append({
                    "id": submission.id,
                    "title": submission.title,
                    "selftext": submission.selftext,
                    "url": submission.url,
                    "author": str(submission.author) if submission.author else '[deleted]',
                    "created_utc": submission.created_utc,
                    "score": submission.score,
                    "subreddit": subreddit,
                    "num_comments": submission.num_comments
                })
        except Exception as e:
            print(f"Error fetching posts from r/{subreddit}: {e}")
        return posts

    def fetch_comments(self, post_id: str, limit: int = 10) -> List[Dict]:
        comments = []
        try:
            submission = self.reddit.submission(id=post_id)
            submission.comments.replace_more(limit=0)
            for comment in submission.comments.list()[:limit]:
                comments.append({
                    "body": comment.body,
                    "score": comment.score,
                    "id": comment.id,
                    "created_utc": comment.created_utc
                })
        except Exception as e:
            print(f"Error fetching comments for {post_id}: {e}")
        return comments