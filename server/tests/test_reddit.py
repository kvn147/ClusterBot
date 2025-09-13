import os
import pytest
from unittest.mock import patch, MagicMock
from app.reddit_client import RedditClient


@pytest.mark.skipif(
    "REDDIT_CLIENT_ID" not in os.environ, reason="Reddit credentials not available"
)
def test_reddit_connection():
    client = RedditClient()
    assert client.reddit is not None


# Alternative test with mocking
def test_reddit_connection_mocked():
    with patch("app.reddit_client.praw.Reddit") as mock_reddit:
        mock_reddit.return_value = MagicMock()
        client = RedditClient()
        assert client.reddit is not None
