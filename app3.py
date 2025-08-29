import streamlit as st
import uuid
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
import time
import json
import random
import math
from io import BytesIO
from PIL import Image, ImageOps, ImageEnhance
import base64
import requests

# -----------------------------
# THIRD-PARTY IMPORTS
# -----------------------------
try:
    from better_profanity import profanity
    PROFANITY_OK = True
except ImportError:
    PROFANITY_OK = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# -----------------------------
# APP CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="SocialVerse - Next Gen Social Media",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/socialverse',
        'Report a bug': "https://github.com/yourusername/socialverse/issues",
        'About': "# SocialVerse - The next generation social media platform"
    }
)

# -----------------------------
# CONSTANTS & PATHS
# -----------------------------
APP_DIR = Path.cwd()
UPLOAD_DIR = APP_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Supported media formats
IMAGE_EXTS = ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp']
VIDEO_EXTS = ['.mp4', '.mov', '.avi', '.webm', '.mkv']
AUDIO_EXTS = ['.mp3', '.wav', '.ogg', '.m4a']

# -----------------------------
# ADVANCED THEME & GLOBAL CSS
# -----------------------------
st.markdown("""
<style>
    :root {
        --primary-gradient: linear-gradient(135deg, #6A11CB 0%, #2575FC 100%);
        --secondary-gradient: linear-gradient(135deg, #FF057C 0%, #8D0B93 100%);
        --dark-bg: #121212;
        --darker-bg: #0A0A0A;
        --card-dark: #1E1E1E;
        --card-light: #FFFFFF;
        --text-dark: #FFFFFF;
        --text-light: #333333;
        --accent: #FF057C;
        --success: #00C853;
        --warning: #FFAB00;
        --error: #FF1744;
        --gray-1: #F5F5F5;
        --gray-2: #EEEEEE;
        --gray-3: #E0E0E0;
        --gray-7: #616161;
        --gray-8: #424242;
        --gray-9: #212121;
    }

    .main { 
        background-color: var(--dark-bg); 
        color: var(--text-dark);
        transition: all 0.3s ease;
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: var(--gray-9);
    }
    ::-webkit-scrollbar-thumb {
        background: var(--accent);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #FF389B;
    }

    /* Buttons */
    .stButton > button {
        background: var(--primary-gradient);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.2s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        background: linear-gradient(135deg, #2575FC 0%, #6A11CB 100%);
    }

    /* Input fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: var(--card-dark);
        color: var(--text-dark);
        border: 1px solid var(--gray-8);
        border-radius: 12px;
        padding: 12px;
    }

    /* File uploader */
    .stFileUploader > div {
        background-color: var(--card-dark);
        border: 2px dashed var(--gray-8);
        border-radius: 12px;
        padding: 20px;
    }

    /* Cards */
    .post-card {
        background-color: var(--card-dark);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        transition: transform 0.2s ease;
        border: 1px solid var(--gray-8);
    }
    .post-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.2);
    }

    /* Stories */
    .story-circle {
        width: 70px;
        height: 70px;
        border-radius: 50%;
        padding: 3px;
        background: var(--secondary-gradient);
        margin: 0 auto;
        cursor: pointer;
        transition: transform 0.2s ease;
    }
    .story-circle:hover {
        transform: scale(1.1);
    }
    .story-circle-inner {
        width: 100%;
        height: 100%;
        border-radius: 50%;
        overflow: hidden;
        border: 2px solid var(--dark-bg);
        background: var(--card-dark);
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .story-username {
        text-align: center;
        margin-top: 8px;
        font-size: 0.8rem;
        max-width: 70px;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* Metrics */
    .metrics {
        display: flex;
        justify-content: space-between;
        margin-top: 15px;
        color: var(--gray-7);
        font-size: 0.9rem;
    }

    /* Timestamp */
    .timestamp {
        font-size: 0.75rem;
        color: var(--gray-7);
        margin-top: 5px;
    }

    /* For you / Following tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: var(--card-dark);
        border-radius: 12px;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: var(--primary-gradient);
        color: white;
    }

    /* Notification badge */
    .notification-badge {
        position: absolute;
        top: -5px;
        right: -5px;
        background: var(--error);
        color: white;
        border-radius: 50%;
        width: 18px;
        height: 18px;
        font-size: 0.7rem;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    /* Swipeable carousel */
    .carousel-container {
        position: relative;
        overflow: hidden;
        border-radius: 16px;
    }
    .carousel-track {
        display: flex;
        transition: transform 0.5s ease;
    }
    .carousel-item {
        flex: 0 0 auto;
        width: 100%;
    }
    .carousel-nav {
        position: absolute;
        top: 50%;
        transform: translateY(-50%);
        background: rgba(0, 0, 0, 0.5);
        color: white;
        border: none;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        z-index: 10;
    }
    .carousel-nav.prev {
        left: 10px;
    }
    .carousel-nav.next {
        right: 10px;
    }

    /* Video container for TikTok-like feed */
    .video-container {
        position: relative;
        width: 100%;
        height: 80vh;
        background: black;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
    }

    /* Marketplace product card */
    .product-card {
        position: relative;
        width: 100%;
        height: 80vh;
        background: var(--card-dark);
        border-radius: 16px;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    .product-media {
        width: 100%;
        height: 70%;
        object-fit: cover;
    }
    .product-info {
        padding: 10px;
        color: var(--text-dark);
    }
    .product-actions {
        display: flex;
        justify-content: space-between;
        padding: 10px;
        background: rgba(0, 0, 0, 0.7);
    }
</style>
""", unsafe_allow_html=True)

# Apply theme to body
st.session_state.setdefault("theme", "dark")
st.markdown(f'<body data-theme="{st.session_state.theme}"></body>', unsafe_allow_html=True)

# -----------------------------
# ADVANCED UTILITIES
# -----------------------------
ISO_FMT = "%Y-%m-%d %H:%M:%S"

def now_str() -> str:
    """Get current timestamp as string."""
    return datetime.now().strftime(ISO_FMT)

def format_timestamp(timestamp_str: str) -> str:
    """Convert timestamp to human-readable format."""
    try:
        timestamp = datetime.strptime(timestamp_str, ISO_FMT)
        now = datetime.now()
        diff = now - timestamp
        
        if diff.days > 365:
            return f"{diff.days // 365}y ago"
        elif diff.days > 30:
            return f"{diff.days // 30}mo ago"
        elif diff.days > 0:
            return f"{diff.days}d ago"
        elif diff.seconds > 3600:
            return f"{diff.seconds // 3600}h ago"
        elif diff.seconds > 60:
            return f"{diff.seconds // 60}m ago"
        else:
            return "Just now"
    except:
        return timestamp_str

SAFE_FILENAME_RE = re.compile(r"[^a-zA-Z0-9_.-]", re.IGNORECASE)

def safe_filename(name: str) -> str:
    """Clean filename for safe storage."""
    name = SAFE_FILENAME_RE.sub("_", name)
    return name[:120]

def generate_thumbnail(video_path: str, output_path: str, time_sec: int = 5) -> bool:
    """Generate a thumbnail from a video."""
    if not CV2_AVAILABLE:
        return False
        
    try:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
        success, image = cap.read()
        if success:
            cv2.imwrite(output_path, image)
            return True
    except Exception as e:
        print(f"Error generating thumbnail: {e}")
    return False

def get_video_duration(video_path: str) -> float:
    """Get the duration of a video in seconds."""
    if not CV2_AVAILABLE:
        return 0.0
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0.0
        cap.release()
        return duration
    except Exception as e:
        print(f"Error getting video duration: {e}")
        return 0.0

def process_image(image_path: str, output_path: str, max_size: Tuple[int, int] = (1080, 1080)) -> bool:
    """Process and optimize an image."""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            # Apply aspect ratio preserving resize
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Enhance image quality
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.2)
            
            # Save optimized image
            img.save(output_path, 'JPEG' if output_path.lower().endswith('.jpg') else 'PNG', 
                    optimize=True, quality=85)
            return True
    except Exception as e:
        print(f"Error processing image: {e}")
        return False

def save_uploaded_file(uploaded_file, subdir: str = "", optimize: bool = True) -> Optional[str]:
    """Save uploaded file to disk and return path."""
    if uploaded_file is None:
        return None
    
    dest_dir = UPLOAD_DIR / subdir
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = safe_filename(f"{ts}_{uploaded_file.name}")
    dest_path = dest_dir / filename
    
    try:
        with dest_path.open("wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Optimize images if requested
        if optimize and any(dest_path.suffix.lower() == ext for ext in IMAGE_EXTS):
            optimized_path = dest_dir / f"optimized_{filename}"
            if process_image(str(dest_path), str(optimized_path)):
                dest_path.unlink()  # Remove original
                return str(optimized_path)
        
        # Generate thumbnails for videos
        if any(dest_path.suffix.lower() == ext for ext in VIDEO_EXTS):
            thumbnail_path = dest_dir / f"thumb_{filename}.jpg"
            if generate_thumbnail(str(dest_path), str(thumbnail_path)):
                pass  # Thumbnail generated successfully
        
        return str(dest_path)
    except Exception as e:
        st.error(f"Failed to save file: {e}")
        return None

def get_file_type(filename: str) -> str:
    """Determine file type from extension."""
    ext = Path(filename).suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    elif ext in VIDEO_EXTS:
        return "video"
    elif ext in AUDIO_EXTS:
        return "audio"
    else:
        return "other"

# -----------------------------
# ADVANCED MODERATION
# -----------------------------
if PROFANITY_OK:
    profanity.load_censor_words()

BLOCK_RE = re.compile(r"(sex|nude|xxx|explicit|porn|violence|hate|racist)", re.IGNORECASE)

def moderate_text(text: Optional[str]) -> bool:
    """Check if text content is appropriate."""
    if not text or not isinstance(text, str):
        return False
    
    # Basic length check
    if len(text.strip()) < 2:  # At least 2 characters
        return False
        
    # Check for excessive repetition (spam detection)
    if len(set(text)) < 5 and len(text) > 20:
        return False
        
    if PROFANITY_OK and profanity.contains_profanity(text):
        return False
    if BLOCK_RE.search(text):
        return False
        
    return True

def moderate_media(filename: Optional[str]) -> bool:
    """Check if media filename is appropriate."""
    if not filename:
        return True
    lower = filename.lower()
    banned_terms = ("porn", "nude", "xxx", "explicit", "violence", "hate")
    return not any(t in lower for t in banned_terms)

# -----------------------------
# AI-ENHANCED FEATURES
# -----------------------------
def generate_hashtags(text: str) -> List[str]:
    """Generate relevant hashtags from text content."""
    words = text.lower().split()
    common_words = {"the", "and", "or", "but", "a", "an", "in", "on", "at", "to", "for", "of", "with"}
    
    hashtags = []
    for word in words:
        if (len(word) > 3 and word not in common_words and 
            not word.startswith(('#', '@')) and word.isalpha()):
            hashtags.append(f"#{word}")
    
    trending = ["#trending", "#viral", "#fyp", "#foryou", "#marketplace", "#shop"]
    if len(hashtags) < 3 and random.random() > 0.7:
        hashtags.extend(random.sample(trending, 2))
    
    return list(set(hashtags))[:5]

def analyze_sentiment(text: str) -> str:
    """Simple sentiment analysis (placeholder for real AI)."""
    positive_words = {"good", "great", "awesome", "amazing", "love", "like", "happy", "nice", "beautiful"}
    negative_words = {"bad", "awful", "hate", "terrible", "dislike", "sad", "angry", "ugly"}
    
    words = set(text.lower().split())
    pos_count = len(words.intersection(positive_words))
    neg_count = len(words.intersection(negative_words))
    
    if pos_count > neg_count:
        return "positive"
    elif neg_count > pos_count:
        return "negative"
    else:
        return "neutral"

# -----------------------------
# STATE INITIALIZATION
# -----------------------------
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "username": "",
        "user_profile": {
            "bio": "",
            "avatar": None,
            "followers": 0,
            "following": 0,
            "posts": 0
        },
        "chat_requests": defaultdict(list),
        "active_chats": defaultdict(lambda: defaultdict(lambda: deque(maxlen=200))),
        "feed_posts": deque(maxlen=500),
        "stories": defaultdict(lambda: deque(maxlen=24)),
        "notifications": deque(maxlen=100),
        "theme": "dark",
        "viewed_stories": set(),
        "follows": set(),
        "blocked": set(),
        "liked_posts": set(),
        "saved_posts": set(),
        "fyp_algorithm": "popular",
        "current_story_view": None,
        "hashtags_followed": set(),
        "trending_hashtags": {},
        "last_notification_check": datetime.now().timestamp(),
        "current_marketplace_index": 0
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# -----------------------------
# ADVANCED CHAT SYSTEM
# -----------------------------
def request_chat(from_user: str, to_user: str) -> bool:
    """Send a chat request from one user to another."""
    if (not to_user or from_user == to_user or 
        to_user in st.session_state.blocked or
        from_user in st.session_state.blocked):
        return False
    
    if to_user not in st.session_state.chat_requests:
        st.session_state.chat_requests[to_user] = []
    
    if from_user in st.session_state.chat_requests[to_user]:
        return False
    
    st.session_state.chat_requests[to_user].append(from_user)
    
    add_notification(to_user, f"{from_user} wants to chat with you", "chat_request")
    return True

def accept_request(from_user: str, to_user: str) -> bool:
    """Accept a chat request and create active chat."""
    reqs = st.session_state.chat_requests.get(to_user, [])
    if from_user in reqs:
        reqs.remove(from_user)
        _ = st.session_state.active_chats[to_user][from_user]
        _ = st.session_state.active_chats[from_user][to_user]
        
        add_notification(from_user, f"{to_user} accepted your chat request", "chat_accepted")
        return True
    return False

def reject_request(from_user: str, to_user: str) -> bool:
    """Reject a chat request."""
    reqs = st.session_state.chat_requests.get(to_user, [])
    if from_user in reqs:
        reqs.remove(from_user)
        return True
    return False

def send_message(from_user: str, to_user: str, msg: str, msg_type: str = "text") -> bool:
    """Send a message between two users."""
    if not moderate_text(msg) and msg_type == "text":
        return False
    if to_user not in st.session_state.active_chats.get(from_user, {}):
        return False
    
    message = {
        "id": str(uuid.uuid4()),
        "from": from_user, 
        "text": msg.strip(), 
        "type": msg_type,
        "timestamp": now_str(),
        "read": False
    }
    
    st.session_state.active_chats[from_user][to_user].append(message)
    st.session_state.active_chats[to_user][from_user].append(message)
    
    if st.session_state.get("current_chat_peer") != to_user:
        add_notification(to_user, f"New message from {from_user}", "new_message")
    return True

def get_unread_count(user: str) -> int:
    """Get count of unread messages for a user."""
    count = 0
    for peer, messages in st.session_state.active_chats.get(user, {}).items():
        for msg in messages:
            if msg["from"] != user and not msg["read"]:
                count += 1
    return count

def mark_as_read(user: str, peer: str):
    """Mark all messages from a peer as read."""
    if peer in st.session_state.active_chats.get(user, {}):
        for msg in st.session_state.active_chats[user][peer]:
            if msg["from"] != user:
                msg["read"] = True

# -----------------------------
# ADVANCED FEED SYSTEM
# -----------------------------
def create_post(user: str, text: str, media_file, is_sell: bool = False, 
               location: str = None, hashtags: List[str] = None, 
               title: str = None, category: str = None, condition: str = None) -> bool:
    """Create a new feed post."""
    if not moderate_text(text):
        return False
    
    media_path = None
    if media_file is not None:
        media_path = save_uploaded_file(media_file, subdir="feed")
        if not moderate_media(media_path):
            return False
    
    if not hashtags and text:
        hashtags = generate_hashtags(text)
    
    post = {
        "id": str(uuid.uuid4()),
        "user": user,
        "text": text.strip(),
        "media_path": media_path,
        "media_type": get_file_type(media_path) if media_path else None,
        "likes": 0,
        "comments": [],
        "shares": 0,
        "views": 0,
        "is_sell": bool(is_sell),
        "location": location,
        "hashtags": hashtags or [],
        "sentiment": analyze_sentiment(text),
        "timestamp": now_str(),
        "price": None,
        "promoted": False,
        "is_reel": False,
        "duration": 0.0,
        "title": title,
        "category": category,
        "condition": condition
    }
    
    if post["media_type"] == "video":
        post["duration"] = get_video_duration(media_path)
        if post["duration"] > 0 and post["duration"] < 60:
            post["is_reel"] = True
    
    if is_sell:
        price_match = re.search(r'\$\d+(?:\.\d{2})?', text)
        if price_match:
            post['price'] = price_match.group(0)
    
    st.session_state.feed_posts.appendleft(post)
    
    st.session_state.user_profile["posts"] += 1
    
    for tag in post['hashtags']:
        st.session_state.trending_hashtags[tag] = st.session_state.trending_hashtags.get(tag, 0) + 1
    
    return True

def like_post(post_id: str, user: str) -> bool:
    """Like a post by ID."""
    for post in st.session_state.feed_posts:
        if post["id"] == post_id:
            if (post_id, user) in st.session_state.liked_posts:
                post["likes"] -= 1
                st.session_state.liked_posts.remove((post_id, user))
                if post["user"] != user:
                    add_notification(post["user"], f"{user} unliked your post", "unlike")
            else:
                post["likes"] += 1
                st.session_state.liked_posts.add((post_id, user))
                if post["user"] != user:
                    add_notification(post["user"], f"{user} liked your post", "like")
            return True
    return False

def comment_post(post_id: str, user: str, comment: str) -> bool:
    """Add a comment to a post."""
    if not moderate_text(comment):
        return False
    
    for post in st.session_state.feed_posts:
        if post["id"] == post_id:
            post["comments"].append({
                "id": str(uuid.uuid4()),
                "user": user,
                "text": comment.strip(),
                "timestamp": now_str(),
                "likes": 0
            })
            if post["user"] != user:
                add_notification(post["user"], f"{user} commented on your post", "comment")
            return True
    return False

def share_post(post_id: str, user: str) -> bool:
    """Share a post."""
    for post in st.session_state.feed_posts:
        if post["id"] == post_id:
            post["shares"] += 1
            if post["user"] != user:
                add_notification(post["user"], f"{user} shared your post", "share")
            return True
    return False

def view_post(post_id: str) -> bool:
    """Increment post view count."""
    for post in st.session_state.feed_posts:
        if post["id"] == post_id:
            post["views"] += 1
            return True
    return False

def delete_post(post_id: str, user: str) -> bool:
    """Delete a post (only by the author)."""
    for post in list(st.session_state.feed_posts):
        if post["id"] == post_id and post["user"] == user:
            st.session_state.feed_posts.remove(post)
            media_path = post.get("media_path")
            if media_path:
                try:
                    Path(media_path).unlink(missing_ok=True)
                except Exception:
                    pass
            st.session_state.user_profile["posts"] -= 1
            return True
    return False

def save_post(post_id: str, user: str) -> bool:
    """Save a post to user's saved items."""
    if (post_id, user) in st.session_state.saved_posts:
        st.session_state.saved_posts.remove((post_id, user))
        return False
    else:
        st.session_state.saved_posts.add((post_id, user))
        return True

# -----------------------------
# STORIES SYSTEM
# -----------------------------
def create_story(user: str, media_file, text: str = None, duration: int = 24) -> bool:
    """Create a new story."""
    if not media_file:
        return False
    
    media_path = save_uploaded_file(media_file, subdir="stories")
    if not moderate_media(media_path):
        return False
    
    story = {
        "id": str(uuid.uuid4()),
        "user": user,
        "media_path": media_path,
        "media_type": get_file_type(media_path),
        "text": text,
        "timestamp": now_str(),
        "expires_at": (datetime.now() + timedelta(hours=duration)).strftime(ISO_FMT),
        "views": 0
    }
    
    st.session_state.stories[user].append(story)
    return True

def view_story(story_id: str, user: str) -> bool:
    """Mark a story as viewed by a user."""
    for username, stories in st.session_state.stories.items():
        for story in stories:
            if story["id"] == story_id:
                if f"{story_id}_{user}" not in st.session_state.viewed_stories:
                    story["views"] += 1
                    st.session_state.viewed_stories.add(f"{story_id}_{user}")
                    return True
    return False

def get_active_stories() -> Dict:
    """Get all active stories."""
    active_stories = {}
    now = datetime.now()
    
    for user, stories in st.session_state.stories.items():
        active = []
        for story in stories:
            expires = datetime.strptime(story["expires_at"], ISO_FMT)
            if expires > now:
                active.append(story)
        
        if active:
            active_stories[user] = active
    
    return active_stories

# -----------------------------
# NOTIFICATION SYSTEM
# -----------------------------
def add_notification(user: str, message: str, notif_type: str = "info"):
    """Add a notification for a user."""
    notification = {
        "id": str(uuid.uuid4()),
        "message": message,
        "type": notif_type,
        "timestamp": now_str(),
        "read": False
    }
    
    st.session_state.notifications.append(notification)

def get_unread_notifications(user: str) -> List:
    """Get unread notifications for a user."""
    return [n for n in st.session_state.notifications if not n["read"]]

def mark_notifications_read():
    """Mark all notifications as read."""
    for notification in st.session_state.notifications:
        notification["read"] = True

# -----------------------------
# FOLLOW SYSTEM
# -----------------------------
def follow_user(user: str, to_follow: str) -> bool:
    """Follow a user."""
    if user == to_follow:
        return False
    
    if (user, to_follow) in st.session_state.follows:
        st.session_state.follows.remove((user, to_follow))
        st.session_state.user_profile["following"] -= 1
        add_notification(to_follow, f"{user} unfollowed you", "unfollow")
        return False
    else:
        st.session_state.follows.add((user, to_follow))
        st.session_state.user_profile["following"] += 1
        add_notification(to_follow, f"{user} started following you", "follow")
        return True

def is_following(user: str, target: str) -> bool:
    """Check if a user is following another."""
    return (user, target) in st.session_state.follows

def get_followers(user: str) -> List[str]:
    """Get list of followers for a user."""
    return [f[0] for f in st.session_state.follows if f[1] == user]

def get_following(user: str) -> List[str]:
    """Get list of users followed by a user."""
    return [f[1] for f in st.session_state.follows if f[0] == user]

# -----------------------------
# ADVANCED UI COMPONENTS
# -----------------------------
def toggle_theme():
    """Toggle between light and dark themes."""
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
    st.rerun()

def render_login():
    """Render login sidebar."""
    with st.sidebar:
        st.title("👤 User Profile")
        
        avatar_col1, avatar_col2 = st.columns([1, 2])
        with avatar_col1:
            avatar = st.file_uploader("Upload avatar", type=IMAGE_EXTS, key="avatar_upload")
            if avatar:
                avatar_path = save_uploaded_file(avatar, subdir="avatars")
                if avatar_path:
                    st.session_state.user_profile["avatar"] = avatar_path
        
        with avatar_col2:
            username = st.text_input(
                "Username",
                value=st.session_state.get("username", ""),
                key="username_input",
                help="Choose your username"
            )
            
            if st.button("Set Username", key="set_username_btn"):
                clean = username.strip()
                if clean:
                    st.session_state.username = clean
                    st.success(f"Welcome, {clean}!")
                else:
                    st.error("Username cannot be empty.")
        
        bio = st.text_area("Bio", key="bio_input", 
                          value=st.session_state.user_profile.get("bio", ""),
                          placeholder="Tell everyone about yourself")
        if bio:
            st.session_state.user_profile["bio"] = bio
        
        if st.session_state.username:
            col1, col2, col3 = st.columns(3)
            col1.metric("Posts", st.session_state.user_profile["posts"])
            col2.metric("Followers", st.session_state.user_profile["followers"])
            col3.metric("Following", st.session_state.user_profile["following"])
        
        st.markdown("---")
        
        st.title("🎨 Theme")
        mode_label = "Light" if st.session_state.theme == "dark" else "Dark"
        if st.button(f"Switch to {mode_label} Mode", key="toggle_theme_btn"):
            toggle_theme()
        
        st.markdown("---")
        st.title("🔔 Notifications")
        unread = len(get_unread_notifications(st.session_state.username))
        if unread > 0:
            st.markdown(f'<span style="color: var(--accent); font-weight: bold;">{unread} unread</span>', 
                       unsafe_allow_html=True)
        
        if st.button("Mark all as read", key="mark_read_btn"):
            mark_notifications_read()
            st.rerun()
        
        for notif in list(st.session_state.notifications)[-5:]:
            st.caption(f"{notif['message']} - {format_timestamp(notif['timestamp'])}")

def ensure_logged_in() -> bool:
    """Check if user is logged in."""
    if not st.session_state.get("username"):
        st.warning("Please enter a username in the sidebar to use the app.")
        return False
    return True

def render_stories():
    """Render stories carousel."""
    active_stories = get_active_stories()
    if not active_stories:
        return
    
    st.markdown("### 📸 Stories")
    
    cols = st.columns(min(8, len(active_stories) + 1))
    
    for i, (user, stories) in enumerate(list(active_stories.items())[:8]):
        with cols[i]:
            has_unseen = any(f"{s['id']}_{st.session_state.username}" not in 
                           st.session_state.viewed_stories for s in stories)
            
            border_style = "var(--secondary-gradient)" if has_unseen else "var(--gray-7)"
            st.markdown(f"""
            <div class="story-circle" style="background: {border_style};">
                <div class="story-circle-inner">
                    <img src="https://via.placeholder.com/64" width="64" height="64" style="object-fit: cover;">
                </div>
            </div>
            <div class="story-username">{user}</div>
            """, unsafe_allow_html=True)
            
            if st.button("View", key=f"story_{user}", use_container_width=True):
                st.session_state.current_story_view = user

def render_post_media(media_path: Optional[str], media_type: Optional[str], post_id: str, is_marketplace: bool = False):
    """Render media content for posts."""
    if not media_path or not Path(media_path).exists():
        st.caption("Media file not found")
        return
    
    view_post(post_id)
    
    try:
        if media_type == "image":
            st.image(media_path, use_column_width=True)
        elif media_type == "video":
            with open(media_path, "rb") as f:
                video_bytes = f.read()
            video_base64 = base64.b64encode(video_bytes).decode('utf-8')
            st.markdown(f'''
            <video controls autoplay muted loop playsinline class="{'product-media' if is_marketplace else 'video-container'}">
            <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
            </video>
            ''', unsafe_allow_html=True)
        elif media_type == "audio":
            st.audio(media_path)
        else:
            st.caption(f"📎 Attachment: {Path(media_path).name}")
    except Exception as e:
        st.caption(f"Could not display media: {e}")

def render_post(post, show_actions=True, is_marketplace=False):
    """Render a single post with all interactions."""
    with st.container():
        if is_marketplace:
            st.markdown('<div class="product-card">', unsafe_allow_html=True)
        else:
            st.markdown('<div class="post-card">', unsafe_allow_html=True)
        
        if is_marketplace:
            if post.get('media_path'):
                render_post_media(post['media_path'], post.get('media_type'), post['id'], is_marketplace=True)
            
            st.markdown('<div class="product-info">', unsafe_allow_html=True)
            st.markdown(f"**{post.get('title', 'Untitled')}**")
            st.markdown(f"Price: {post.get('price', 'N/A')}")
            st.caption(f"Category: {post.get('category', 'N/A')} | Condition: {post.get('condition', 'N/A')}")
            st.write(post['text'])
            if post.get('location'):
                st.caption(f"📍 {post['location']}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="product-actions">', unsafe_allow_html=True)
            if st.button("Contact Seller", key=f"contact_{post['id']}"):
                if request_chat(st.session_state.username, post['user']):
                    st.session_state.current_chat_peer = post['user']
                    mark_as_read(st.session_state.username, post['user'])
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        
        else:
            col1, col2 = st.columns([1, 10])
            with col1:
                st.image("https://via.placeholder.com/40", width=40)
            
            with col2:
                st.markdown(f"**{post['user']}** · {format_timestamp(post['timestamp'])}")
                if post.get('location'):
                    st.caption(f"📍 {post['location']}")
            
            if post['text']:
                st.write(post['text'])
            
            if post.get('media_path'):
                render_post_media(post['media_path'], post.get('media_type'), post['id'])
            
            if post.get('hashtags'):
                hashtag_text = " ".join(post['hashtags'])
                st.caption(hashtag_text)
            
            if post.get('is_sell') and post.get('price'):
                st.markdown(f"**Price:** {post['price']}")
        
        if show_actions and not is_marketplace:
            col1, col2, col3, col4, col5 = st.columns(5)
            
            is_liked = (post['id'], st.session_state.username) in st.session_state.liked_posts
            like_icon = "❤️" if is_liked else "🤍"
            if col1.button(f"{like_icon} {post['likes']}", key=f"like_{post['id']}"):
                like_post(post['id'], st.session_state.username)
                st.rerun()
            
            if col2.button(f"💬 {len(post['comments'])}", key=f"comment_btn_{post['id']}"):
                st.session_state[f"show_comments_{post['id']}"] = not st.session_state.get(f"show_comments_{post['id']}")
                st.rerun()
            
            if col3.button(f"↗️ {post['shares']}", key=f"share_{post['id']}"):
                share_post(post['id'], st.session_state.username)
                st.success("Post shared!")
            
            is_saved = (post['id'], st.session_state.username) in st.session_state.saved_posts
            save_icon = "🔖" if is_saved else "📑"
            if col4.button(save_icon, key=f"save_{post['id']}"):
                save_post(post['id'], st.session_state.username)
                st.rerun()
            
            with col5:
                if post['user'] == st.session_state.username:
                    if st.button("🗑️", key=f"delete_opt_{post['id']}"):
                        if delete_post(post['id'], st.session_state.username):
                            st.success("Post deleted!")
                            st.rerun()
                else:
                    if st.button("⋯", key=f"more_{post['id']}"):
                        pass
        
        if not is_marketplace:
            st.markdown(f"""
            <div class="metrics">
                <span>{post['views']} views</span>
                <span>{post['likes']} likes</span>
                <span>{len(post['comments'])} comments</span>
                <span>{post['shares']} shares</span>
            </div>
            """, unsafe_allow_html=True)
        
        if st.session_state.get(f"show_comments_{post['id']}") and not is_marketplace:
            st.markdown("---")
            st.markdown("**Comments**")
            
            for comment in post['comments']:
                st.caption(f"**{comment['user']}**: {comment['text']} · {format_timestamp(comment['timestamp'])}")
            
            with st.form(key=f"comment_form_{post['id']}"):
                comment_text = st.text_input("Add a comment:", key=f"comment_input_{post['id']}", 
                                           placeholder="Write a comment...")
                if st.form_submit_button("Post Comment"):
                    if comment_text.strip():
                        if comment_post(post['id'], st.session_state.username, comment_text.strip()):
                            st.rerun()
                        else:
                            st.error("Comment contains inappropriate content.")
        
        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# PAGES
# -----------------------------
def render_feed_page():
    """Render the main feed page."""
    st.title("🌐 SocialVerse Feed")
    
    render_stories()
    
    with st.expander("Create a New Post", expanded=False):
        with st.form("create_post_form", clear_on_submit=True):
            text = st.text_area("What's happening?", key="post_text_input", 
                               height=100, placeholder="Share your thoughts...")
            
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                media = st.file_uploader("Add media (video for reels recommended)", 
                                       type=IMAGE_EXTS + VIDEO_EXTS + AUDIO_EXTS,
                                       key="post_media_uploader")
            with col2:
                location = st.text_input("Location", key="post_location_input",
                                       placeholder="Add location")
            with col3:
                is_sell = st.checkbox("For Sale", key="is_sell_checkbox")
            
            submitted = st.form_submit_button("📝 Post", use_container_width=True)
            
        if submitted:
            if (text or "").strip() or media:
                if create_post(st.session_state.username, text.strip() if text else "", 
                              media, is_sell, location):
                    st.success("Posted successfully!")
                    st.rerun()
                else:
                    st.error("Your post may contain inappropriate content or failed to upload.")
            else:
                st.error("Post must contain text or media.")

    tab1, tab2, tab3 = st.tabs(["For You", "Following", "Reels"])
    
    with tab1:
        st.subheader("For You")
        if not st.session_state.feed_posts:
            st.info("No posts yet. Be the first to share something!")
        else:
            for post in list(st.session_state.feed_posts)[:20]:
                render_post(post)
    
    with tab2:
        st.subheader("Following")
        following = get_following(st.session_state.username)
        if not following:
            st.info("You're not following anyone yet. Follow users to see their posts here.")
        else:
            following_posts = [p for p in st.session_state.feed_posts if p['user'] in following]
            if not following_posts:
                st.info("No posts from people you follow yet.")
            for post in following_posts[:20]:
                render_post(post)
    
    with tab3:
        st.subheader("Reels")
        if 'current_reel_index' not in st.session_state:
            st.session_state.current_reel_index = 0
        
        video_posts = [p for p in st.session_state.feed_posts if p.get('is_reel', False)]
        
        if not video_posts:
            st.info("No reels yet. Upload short videos to create reels!")
        else:
            random.shuffle(video_posts)
            index = st.session_state.current_reel_index % len(video_posts)
            post = video_posts[index]
            
            st.markdown('<div class="video-container">', unsafe_allow_html=True)
            render_post_media(post['media_path'], 'video', post['id'])
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.write(f"**{post['user']}**: {post['text']}")
            
            col1, col2, col3, col4 = st.columns(4)
            is_liked = (post['id'], st.session_state.username) in st.session_state.liked_posts
            like_icon = "❤️" if is_liked else "🤍"
            if col1.button(f"{like_icon} {post['likes']}"):
                like_post(post['id'], st.session_state.username)
                st.rerun()
            
            if col2.button(f"💬 {len(post['comments'])}"):
                st.session_state[f"show_comments_{post['id']}"] = not st.session_state.get(f"show_comments_{post['id']}", False)
                st.rerun()
            
            if col3.button(f"↗️ {post['shares']}"):
                share_post(post['id'], st.session_state.username)
                st.rerun()
            
            is_saved = (post['id'], st.session_state.username) in st.session_state.saved_posts
            save_icon = "🔖" if is_saved else "📑"
            if col4.button(save_icon):
                save_post(post['id'], st.session_state.username)
                st.rerun()
            
            if st.session_state.get(f"show_comments_{post['id']}"):
                for comment in post['comments']:
                    st.caption(f"**{comment['user']}**: {comment['text']}")
                comment_text = st.text_input("Comment:")
                if st.button("Post Comment"):
                    comment_post(post['id'], st.session_state.username, comment_text)
                    st.rerun()
            
            col1, col2 = st.columns(2)
            if col1.button("Previous Reel"):
                st.session_state.current_reel_index = (st.session_state.current_reel_index - 1) % len(video_posts)
                st.rerun()
            if col2.button("Next Reel"):
                st.session_state.current_reel_index = (st.session_state.current_reel_index + 1) % len(video_posts)
                st.rerun()

def render_chat_page():
    """Render the chat interface."""
    st.title("💬 Messages")
    
    unread_count = get_unread_count(st.session_state.username)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Conversations")
        
        with st.expander("+ New Chat", expanded=False):
            peer = st.text_input("Username", key="new_chat_peer")
            if st.button("Start Chat", key="start_chat_btn"):
                if peer and peer != st.session_state.username:
                    if request_chat(st.session_state.username, peer):
                        st.success(f"Chat request sent to {peer}")
                    else:
                        st.error("Could not send request")
        
        incoming = list(st.session_state.chat_requests.get(st.session_state.username, []))
        if incoming:
            st.markdown("**Requests**")
            for req in incoming:
                c1, c2 = st.columns([2, 1])
                c1.write(req)
                if c2.button("Accept", key=f"accept_{req}"):
                    if accept_request(req, st.session_state.username):
                        st.rerun()
        
        st.markdown("**Chats**")
        active = st.session_state.active_chats.get(st.session_state.username, {})
        for peer in active.keys():
            unread = sum(1 for msg in active[peer] if msg["from"] != peer and not msg["read"])
            badge = f" ({unread})" if unread > 0 else ""
            
            if st.button(f"{peer}{badge}", key=f"chat_btn_{peer}", use_container_width=True):
                st.session_state.current_chat_peer = peer
                mark_as_read(st.session_state.username, peer)
    
    with col2:
        if 'current_chat_peer' in st.session_state:
            peer = st.session_state.current_chat_peer
            st.subheader(f"Chat with {peer}")
            
            messages = st.session_state.active_chats[st.session_state.username][peer]
            for msg in messages:
                if msg['from'] == st.session_state.username:
                    st.markdown(f"""
                    <div style='display: flex; justify-content: flex-end; margin: 5px 0;'>
                        <div style='background: var(--primary-gradient); color: white; 
                                    padding: 10px; border-radius: 15px; max-width: 70%;'>
                            {msg['text']}
                            <div class='timestamp'>{format_timestamp(msg['timestamp'])}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style='display: flex; justify-content: flex-start; margin: 5px 0;'>
                        <div style='background: var(--card-dark); color: var(--text-dark); 
                                    padding: 10px; border-radius: 15px; max-width: 70%;'>
                            {msg['text']}
                            <div class='timestamp'>{format_timestamp(msg['timestamp'])}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with st.form(key=f"chat_form_{peer}"):
                msg_text = st.text_input("Message", key=f"chat_input_{peer}", 
                                       placeholder="Type a message...")
                if st.form_submit_button("Send"):
                    if msg_text.strip():
                        if send_message(st.session_state.username, peer, msg_text.strip()):
                            st.rerun()
                        else:
                            st.error("Could not send message")
        else:
            st.info("Select a conversation or start a new one")

def render_sell_page():
    """Render the marketplace/sell page."""
    st.title("🛍️ Marketplace")
    
    with st.expander("List an Item for Sale", expanded=False):
        with st.form("sell_post_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            
            with col1:
                title = st.text_input("Item Title", key="sell_title_input", placeholder="e.g., Vintage Sneakers")
                price = st.text_input("Price", key="sell_price_input", placeholder="$0.00")
                category = st.selectbox("Category", 
                                      ["All", "Electronics", "Clothing", "Home", "Sports", "Beauty", "Toys", "Books", "Other"],
                                      key="sell_category_input")
            
            with col2:
                media = st.file_uploader("Product Media (video recommended)", type=IMAGE_EXTS + VIDEO_EXTS, 
                                       accept_multiple_files=False, key="sell_media_uploader")
                condition = st.selectbox("Condition", ["New", "Like New", "Good", "Fair"], key="sell_condition_input")
                location = st.text_input("Location", key="sell_location_input", placeholder="e.g., New York, NY")
            
            description = st.text_area("Description", height=100, 
                                     placeholder="Describe your item in detail (e.g., brand, features, condition)...",
                                     key="sell_description_input")
            
            submitted = st.form_submit_button("List Item", use_container_width=True)
            
        if submitted:
            if not title.strip() or not price.strip() or not description.strip():
                st.error("Please fill in all required fields (title, price, description).")
            else:
                full_text = f"{description}\n\nPrice: {price}"
                if create_post(st.session_state.username, full_text, media, 
                              is_sell=True, location=location, title=title, 
                              category=category, condition=condition):
                    st.success("Item listed successfully!")
                    st.rerun()
                else:
                    st.error("Could not create listing. Check content for inappropriate material.")
    
    st.subheader("Browse Marketplace")
    sell_posts = [p for p in st.session_state.feed_posts if p.get("is_sell", False)]
    
    if not sell_posts:
        st.info("No items for sale yet.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        category_filter = st.selectbox("Category", ["All", "Electronics", "Clothing", "Home", "Sports", "Beauty", "Toys", "Books", "Other"])
    with col2:
        price_filter = st.selectbox("Price", ["Any", "Under $10", "$10-$50", "$50-$100", "Over $100"])
    with col3:
        condition_filter = st.selectbox("Condition", ["All", "New", "Like New", "Good", "Fair"])
    with col4:
        sort_by = st.selectbox("Sort by", ["Newest", "Price: Low to High", "Price: High to Low", "Most Popular"])
    
    filtered_posts = sell_posts
    if category_filter != "All":
        filtered_posts = [p for p in filtered_posts if p.get('category', '').lower() == category_filter.lower()]
    if condition_filter != "All":
        filtered_posts = [p for p in filtered_posts if p.get('condition', '') == condition_filter]
    if price_filter != "Any":
        if price_filter == "Under $10":
            filtered_posts = [p for p in filtered_posts if p.get('price') and float(p['price'].replace('$', '').replace(',', '')) < 10]
        elif price_filter == "$10-$50":
            filtered_posts = [p for p in filtered_posts if p.get('price') and 10 <= float(p['price'].replace('$', '').replace(',', '')) <= 50]
        elif price_filter == "$50-$100":
            filtered_posts = [p for p in filtered_posts if p.get('price') and 50 < float(p['price'].replace('$', '').replace(',', '')) <= 100]
        else:
            filtered_posts = [p for p in filtered_posts if p.get('price') and float(p['price'].replace('$', '').replace(',', '')) > 100]
    
    if sort_by == "Newest":
        filtered_posts.sort(key=lambda x: x['timestamp'], reverse=True)
    elif sort_by == "Price: Low to High":
        filtered_posts.sort(key=lambda x: float(x.get('price', '0').replace('$', '').replace(',', '')) 
                          if x.get('price') else 0)
    elif sort_by == "Price: High to Low":
        filtered_posts.sort(key=lambda x: float(x.get('price', '0').replace('$', '').replace(',', '')), 
                          reverse=True)
    elif sort_by == "Most Popular":
        filtered_posts.sort(key=lambda x: x['likes'] + len(x['comments']) + x['views'], reverse=True)
    
    if 'current_marketplace_index' not in st.session_state:
        st.session_state.current_marketplace_index = 0
    
    if filtered_posts:
        random.shuffle(filtered_posts)
        index = st.session_state.current_marketplace_index % len(filtered_posts)
        post = filtered_posts[index]
        
        render_post(post, show_actions=False, is_marketplace=True)
        
        col1, col2 = st.columns(2)
        if col1.button("Previous Item"):
            st.session_state.current_marketplace_index = (st.session_state.current_marketplace_index - 1) % len(filtered_posts)
            st.rerun()
        if col2.button("Next Item"):
            st.session_state.current_marketplace_index = (st.session_state.current_marketplace_index + 1) % len(filtered_posts)
            st.rerun()
    else:
        st.info("No items match the selected filters.")

def render_discover_page():
    """Render the discover/explore page."""
    st.title("🔍 Discover")
    
    st.subheader("Trending Now")
    
    trending_hashtags = sorted(st.session_state.trending_hashtags.items(), 
                              key=lambda x: x[1], reverse=True)[:10]
    
    if trending_hashtags:
        cols = st.columns(5)
        for i, (tag, count) in enumerate(trending_hashtags):
            with cols[i % 5]:
                if st.button(f"{tag} ({count})", key=f"discover_{tag}", use_container_width=True):
                    st.session_state.current_hashtag = tag
        st.markdown("---")
    
    st.subheader("People You May Know")
    
    all_users = set()
    for post in st.session_state.feed_posts:
        all_users.add(post['user'])
    
    suggested_users = [u for u in all_users 
                      if u != st.session_state.username 
                      and not is_following(st.session_state.username, u)]
    
    if suggested_users:
        cols = st.columns(min(4, len(suggested_users)))
        for i, user in enumerate(suggested_users[:4]):
            with cols[i]:
                st.image("https://via.placeholder.com/80", width=80)
                st.write(f"**{user}**")
                user_posts = sum(1 for p in st.session_state.feed_posts if p['user'] == user)
                st.caption(f"{user_posts} posts")
                if st.button("Follow", key=f"follow_{user}"):
                    follow_user(st.session_state.username, user)
                    st.success(f"Now following {user}")
                    st.rerun()
        st.markdown("---")
    
    st.subheader("Popular Content")
    
    popular_posts = sorted(st.session_state.feed_posts, 
                          key=lambda x: x['likes'] + len(x['comments']) + x['shares'], 
                          reverse=True)
    
    following = get_following(st.session_state.username)
    explore_posts = [p for p in popular_posts if p['user'] not in following][:9]
    
    if explore_posts:
        cols = st.columns(3)
        for i, post in enumerate(explore_posts):
            with cols[i % 3]:
                if post.get('media_path'):
                    st.image(post['media_path'], use_column_width=True)
                st.caption(f"by {post['user']} · {post['likes']} likes")
        st.markdown("---")
    
    if 'current_hashtag' in st.session_state:
        st.subheader(f"Posts with {st.session_state.current_hashtag}")
        hashtag_posts = [p for p in st.session_state.feed_posts 
                        if st.session_state.current_hashtag in p.get('hashtags', [])]
        
        if hashtag_posts:
            for post in hashtag_posts:
                render_post(post)
        else:
            st.info(f"No posts found with {st.session_state.current_hashtag}")

def render_notifications_page():
    """Render the notifications page."""
    st.title("🔔 Notifications")
    
    tab1, tab2, tab3 = st.tabs(["All", "Unread", "Mentions"])
    
    with tab1:
        if st.session_state.notifications:
            for notif in reversed(st.session_state.notifications):
                icon = "🔵" if not notif['read'] else "⚪"
                st.markdown(f"{icon} **{notif['message']}** · {format_timestamp(notif['timestamp'])}")
        else:
            st.info("No notifications yet")
    
    with tab2:
        unread = [n for n in st.session_state.notifications if not n['read']]
        if unread:
            for notif in reversed(unread):
                st.markdown(f"🔵 **{notif['message']}** · {format_timestamp(notif['timestamp'])}")
        else:
            st.info("No unread notifications")
    
    with tab3:
        mentions = [n for n in st.session_state.notifications if '@' + st.session_state.username in n['message']]
        if mentions:
            for notif in reversed(mentions):
                st.markdown(f"**{notif['message']}** · {format_timestamp(notif['timestamp'])}")
        else:
            st.info("No mentions yet")
    
    if st.button("Mark all as read", key="notifications_mark_read"):
        mark_notifications_read()
        st.rerun()

def render_profile_page():
    """Render the user profile page."""
    if not st.session_state.username:
        st.warning("Please set a username first")
        return
    
    st.title(f"👤 {st.session_state.username}'s Profile")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        avatar = st.session_state.user_profile.get("avatar")
        if avatar and Path(avatar).exists():
            st.image(avatar, width=150)
        else:
            st.image("https://via.placeholder.com/150", width=150)
        
        if st.button("Edit Profile", use_container_width=True):
            st.session_state.editing_profile = True
    
    with col2:
        col21, col22, col23 = st.columns(3)
        col21.metric("Posts", st.session_state.user_profile["posts"])
        col22.metric("Followers", st.session_state.user_profile["followers"])
        col23.metric("Following", st.session_state.user_profile["following"])
        
        st.write(st.session_state.user_profile.get("bio", "No bio yet"))
    
    if st.session_state.get("editing_profile"):
        with st.form("edit_profile_form"):
            new_bio = st.text_area("Bio", value=st.session_state.user_profile.get("bio", ""))
            new_avatar = st.file_uploader("Profile Picture", type=IMAGE_EXTS)
            
            col1, col2 = st.columns(2)
            if col1.form_submit_button("Save Changes"):
                if new_bio:
                    st.session_state.user_profile["bio"] = new_bio
                if new_avatar:
                    avatar_path = save_uploaded_file(new_avatar, subdir="avatars")
                    if avatar_path:
                        st.session_state.user_profile["avatar"] = avatar_path
                st.session_state.editing_profile = False
                st.success("Profile updated!")
                st.rerun()
            if col2.form_submit_button("Cancel"):
                st.session_state.editing_profile = False
                st.rerun()
    
    tab1, tab2, tab3, tab4 = st.tabs(["Posts", "Reels", "Tagged", "Saved"])
    
    with tab1:
        user_posts = [p for p in st.session_state.feed_posts if p['user'] == st.session_state.username]
        if user_posts:
            for post in user_posts:
                render_post(post, show_actions=True)
        else:
            st.info("You haven't posted anything yet")
    
    with tab2:
        user_videos = [p for p in st.session_state.feed_posts 
                      if p['user'] == st.session_state.username and p.get('media_type') == 'video']
        if user_videos:
            for post in user_videos:
                render_post(post, show_actions=True)
        else:
            st.info("You haven't posted any videos yet")
    
    with tab3:
        tagged_posts = [p for p in st.session_state.feed_posts 
                       if '@' + st.session_state.username in p['text']]
        if tagged_posts:
            for post in tagged_posts:
                render_post(post, show_actions=True)
        else:
            st.info("You haven't been tagged in any posts yet")
    
    with tab4:
        saved_post_ids = [pid for (pid, user) in st.session_state.saved_posts 
                         if user == st.session_state.username]
        saved_posts = [p for p in st.session_state.feed_posts if p['id'] in saved_post_ids]
        if saved_posts:
            for post in saved_posts:
                render_post(post, show_actions=True)
        else:
            st.info("You haven't saved any posts yet")

# -----------------------------
# MAIN APP
# -----------------------------
def main():
    """Main application entry point."""
    render_login()
    
    if not ensure_logged_in():
        st.info("👋 Welcome to SocialVerse! Enter a username in the sidebar to get started.")
        return

    page = st.sidebar.selectbox(
        "Navigate",
        ["🌐 Feed", "🔍 Discover", "💬 Messages", "🛍️ Marketplace", "🔔 Notifications", "👤 Profile"],
        key="navigation_selectbox"
    )

    if page == "🌐 Feed":
        render_feed_page()
    elif page == "🔍 Discover":
        render_discover_page()
    elif page == "💬 Messages":
        render_chat_page()
    elif page == "🛍️ Marketplace":
        render_sell_page()
    elif page == "🔔 Notifications":
        render_notifications_page()
    elif page == "👤 Profile":
        render_profile_page()

if __name__ == "__main__":
    main()