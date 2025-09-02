from sqlalchemy import Column, Integer, String, Text, DateTime, Float, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Post(Base):
    __tablename__ = "posts"
    
    id = Column(String, primary_key=True)  # Reddit post ID
    title = Column(String, nullable=False)
    content = Column(Text)
    url = Column(String)
    author = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    reddit_created_utc = Column(Float)
    score = Column(Integer, default=0)
    subreddit = Column(String, nullable=False)
    num_comments = Column(Integer, default=0)
    
    # Clustering fields
    cluster_id = Column(Integer, ForeignKey('clusters.id'))
    processed = Column(Boolean, default=False)
    
    cluster = relationship("Cluster", back_populates="posts")

class Cluster(Base):
    __tablename__ = "clusters"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    representative_post_id = Column(String, ForeignKey('posts.id'))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    post_count = Column(Integer, default=1)
    keywords = Column(Text)  # JSON string of important keywords
    title = Column(String)  # Generated cluster title
    
    posts = relationship("Post", back_populates="cluster")
    representative_post = relationship("Post", foreign_keys=[representative_post_id])