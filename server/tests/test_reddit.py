from app.reddit_client import RedditClient


def test_reddit_connection():
    client = RedditClient()
    posts = client.get_new_posts("worldnews", limit=5)

    for post in posts:
        print(f"Title: {post['title'][:50]}...")
        print(f"URL: {post['url']}")
        print("---")


if __name__ == "__main__":
    test_reddit_connection()
