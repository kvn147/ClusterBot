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
            user_agent="my_reddit_client"
        )

    def fetch_posts(self, subreddit: str, limit: int = 10) -> List[Dict]:
        posts = []
        try:
            subreddit_instance = self.reddit.subreddit(subreddit)
            for submission in subreddit_instance.hot(limit=limit):
                posts.append({
                    "title": submission.title,
                    "url": submission.url,
                    "score": submission.score,
                    "id": submission.id,
                    "created_utc": submission.created_utc
                })
        except Exception as e:
            print(f"An error occurred: {e}")
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
            print(f"An error occurred: {e}")
        return comments