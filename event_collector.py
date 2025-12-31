#!/usr/bin/env python3
"""
Event Collector for AI/ML Events

Background job that runs every 6 hours to collect upcoming events related to:
- AI, Machine Learning, Deep Learning, Computer Vision, Generative AI, Agentic AI, RAG

Data Sources:
1. Tavily web search (primary and only method - requires TAVILY_API_KEY)

Requirements:
- pip install tavily-python python-dateutil
- Set TAVILY_API_KEY environment variable

Output: JSON cache file with deduplicated future events
"""

import os
import json
import datetime
from dateutil.parser import parse as parse_date, ParserError
from dateutil.tz import tzlocal
import re
import time
from typing import List, Dict, Any, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Configuration
CACHE_FILE = "events_cache.json"
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

# Tavily search queries for AI/ML events (expanded for maximum results)
TAVILY_QUERIES = [
    # Hackathon queries
    "AI hackathon 2026 india kaggle hackerearth",
    "machine learning hackathon 2026 india",
    "deep learning hackathon 2026 india",
    "computer vision hackathon 2026 india",

    # Conference queries
    "AI conference 2026 india neurips icml cvpr aaai",
    "machine learning conference 2026 india",
    "deep learning conference 2026 india",
    "computer vision conference 2026 india",

    # Workshop queries
    "AI workshop 2026 india",
    "machine learning workshop 2026 india",
    "deep learning tutorial 2026 india",
    "computer vision workshop 2026 india",

    # City-specific queries (major tech hubs)
    "AI hackathon 2026 bangalore chennai delhi mumbai",
    "AI conference 2026 bangalore chennai delhi mumbai",
    "AI workshop 2026 bangalore chennai delhi mumbai",

    # Platform-specific queries
    "kaggle competitions machine learning 2026 india",
    "eventbrite AI events 2026 india",
    "meetup AI machine learning 2026 india",

    # Specialized events
    "RAG workshop 2026 india",
    "generative AI conference 2026 india",
    "agentic AI events 2026 india"
]


# Cities for location filtering
TARGET_CITIES = [
    "chennai", "bangalore", "hyderabad", "pune", "noida", "delhi",
    "mumbai", "kolkata", "ahmedabad", "jaipur", "chandigarh", "gurgaon"
]

# Tavily search configuration (balanced for more queries)
TAVILY_MAX_RESULTS = 5  # Results per query (reduced for more queries)
TAVILY_REQUEST_DELAY = 1  # Seconds between requests

# Date patterns for parsing (improved for Tavily content)
DATE_RE = re.compile(
    r"((?:\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\b)[\s\.\-]?\d{1,2}(?:st|nd|rd|th)?(?:,?\s*\d{4})?(?:\s*\d{1,2}:\d{2}(?:\s*[AP]M)?)?(?:\s*UTC)?)"
    r"|(\b\d{4}-\d{2}-\d{2}\b)"
    r"|(\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b)"
    r"|((?:\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\b)\s+\d{4})"
    r"|(\b\d{1,2}\.\d{1,2}\.\d{4}\b)"
    r"|(\b\d{4}\.\d{1,2}\.\d{1,2}\b)",
    flags=re.IGNORECASE,
)


def parse_date_candidates(text: str) -> List[datetime.date]:
    """Find date-like substrings and return parsed datetime.date objects."""
    if not text:
        return []

    candidates = []

    # Handle date ranges like "Dec 20-22, 2025"
    range_match = re.search(
        r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]\.?\s\d{1,2})\s*(?:–|—|-|to)\s*((?:\d{1,2})(?:,?\s*\d{4})?)",
        text,
        flags=re.IGNORECASE,
    )
    if range_match:
        left = range_match.group(1)
        right = range_match.group(2)
        year_search = re.search(r"\b(20\d{2})\b", text)
        try:
            if year_search and not re.search(r"\b\d{4}\b", right):
                right = f"{right}, {year_search.group(1)}"
            start = parse_date(left, fuzzy=True).date()
            end = parse_date(right, fuzzy=True).date()
            candidates.extend([start, end])
            return candidates
        except ParserError:
            pass

    # Parse individual dates with multiple patterns
    date_patterns = [
        # Month Day, Year (Nov 27, 2025)
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b",
        # Month Day Year (Nov 27 2025)
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?\s+\d{4}\b",
        # Month Day (Nov 27)
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?\b",
        # ISO format (2025-11-27)
        r"\b\d{4}-\d{2}-\d{2}\b",
        # DD/MM/YYYY or MM/DD/YYYY
        r"\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4}\b",
        # DD.MM.YYYY
        r"\b\d{1,2}\.\d{1,2}\.\d{4}\b",
    ]

    for pattern in date_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            token = match.group(0)
            if not token:
                continue

            # Skip if it looks like a time (e.g., "03:30")
            if re.match(r'^\d{1,2}:\d{2}', token):
                continue

            try:
                dt = parse_date(token, fuzzy=True)
                candidates.append(dt.date())
            except (ParserError, ValueError):
                continue

    # Handle deadlines and submission dates
    if not candidates:
        deadline_patterns = [
            r"(?:deadline|due|submission|submit|ends|closes|registration)\s*(?:by|on|until)?\s*[:\-]?\s*([A-Za-z0-9\,\s\-\/]+)",
            r"([A-Za-z0-9\,\s\-\/]+)\s*(?:deadline|due|submission|submit|ends|closes)",
        ]
        for pattern in deadline_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                try:
                    dt = parse_date(match.group(1), fuzzy=True)
                    candidates.append(dt.date())
                    break
                except (ParserError, ValueError):
                    continue

    # Look for year + month patterns (e.g., "2025 Nov", "November 2025")
    if not candidates:
        year_month_patterns = [
            r"\b(20\d{2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\b",
            r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+(20\d{2})\b",
        ]
        for pattern in year_month_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                try:
                    dt = parse_date(match.group(0), fuzzy=True)
                    candidates.append(dt.date())
                    break
                except (ParserError, ValueError):
                    continue

    return sorted(list(set(candidates)))


def detect_location(text: str) -> str:
    """Detect location from event text with strict context awareness."""
    if not text:
        return "TBD"

    text_lower = text.lower()

    # Check for online/virtual indicators (highest priority)
    online_keywords = ['online', 'virtual', 'remote', 'webinar', 'zoom', 'teams', 'meet', 'google meet', 'livestream']
    if any(keyword in text_lower for keyword in online_keywords):
        return "Online"

    # All cities to check
    all_cities = TARGET_CITIES + ['mumbai', 'kolkata', 'ahmedabad', 'jaipur', 'chandigarh', 'gurgaon', 'pune', 'hyderabad']

    # Skip title-based detection for web search results - it's unreliable
    # Web content doesn't have clear title boundaries, so we rely on explicit patterns only

    # Step 2: Look for explicit location patterns in the full text
    location_patterns = [
        # Very specific location indicators
        r'\b(?:location|venue|address|site|place)\s*:?\s*([a-zA-Z\s]+?)(?:\s*[,\-\|]|$)',
        r'\b(?:held\s+at|taking\s+place\s+at|located\s+at)\s+([a-zA-Z\s]+?)(?:\s*[,\-\|]|$)',
        r'\b(?:event\s+in|conference\s+in|hackathon\s+in)\s+([a-zA-Z\s]+?)(?:\s*[,\-\|]|$)',

        # Geographic with country
        r'\b([a-zA-Z]+(?:\s+[a-zA-Z]+)*),\s*(?:india|IN)\b',
    ]

    for pattern in location_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            location_candidate = match.group(1).strip().lower()
            # Check if the matched location contains any of our target cities
            for city in all_cities:
                if city in location_candidate:
                    return city.title()

    # Step 3: Ultra-restrictive fallback - only detect location if we have VERY strong evidence
    # This prevents false positives from mentions like "events in Chennai, Bangalore..."

    # Only look in the first 300 characters (title + description) to avoid incidental mentions
    relevant_text = text[:300]
    relevant_lower = relevant_text.lower()

    event_keywords = ['hackathon', 'conference', 'workshop', 'summit', 'competition', 'challenge']

    for city in all_cities:
        if city in relevant_lower:
            # Must match one of these very specific patterns to be considered a real location:
            # 1. "hackathon in Chennai"
            # 2. "Chennai hackathon"
            # 3. "AI hackathon Chennai"
            # 4. "Chennai AI hackathon"
            strict_patterns = [
                r'\b(?:' + '|'.join(event_keywords) + r')\s+(?:in|at|@)\s+' + city + r'\b',
                r'\b' + city + r'\s+(?:' + '|'.join(event_keywords) + r')\b',
                r'\b' + city + r'\s+(?:ai|ml|machine learning|deep learning|computer vision)\s+(?:' + '|'.join(event_keywords) + r')\b',
                r'\b(?:ai|ml|machine learning|deep learning|computer vision)\s+(?:' + '|'.join(event_keywords) + r')\s+' + city + r'\b',
                r'\b(?:ai|ml|machine learning|deep learning|computer vision)\s+' + city + r'\s+(?:' + '|'.join(event_keywords) + r')\b',
            ]

            for pattern in strict_patterns:
                if re.search(pattern, relevant_lower):
                    return city.title()

    # NO fallback detection - if we can't find strong evidence, return TBD
    # This prevents false positives from general mentions

    return "TBD"


def detect_event_type(text: str) -> str:
    """Detect if event is online or offline."""
    if not text:
        return "TBD"

    text_lower = text.lower()

    # Online indicators
    online_keywords = ['online', 'virtual', 'remote', 'webinar', 'zoom', 'teams', 'meet', 'google meet', 'livestream']
    if any(keyword in text_lower for keyword in online_keywords):
        return "Online"

    # Offline indicators (mention of physical locations)
    offline_keywords = ['venue', 'hotel', 'center', 'auditorium', 'campus', 'street', 'road']
    if any(keyword in text_lower for keyword in offline_keywords):
        return "Offline"

    # If location is specified but no online keywords, assume offline
    if detect_location(text) != "TBD":
        return "Offline"

    return "TBD"


def search_tavily(query: str) -> List[Dict[str, Any]]:
    """Search for events using Tavily API (token-efficient approach)."""
    if not TAVILY_API_KEY:
        print("[warn] TAVILY_API_KEY not found in environment variables")
        return []

    try:
        from tavily import TavilyClient
    except ImportError:
        print("[warn] tavily-python not available, install with: pip install tavily-python")
        return []

    events = []
    today = datetime.datetime.now(tzlocal()).date()

    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        # Use "advanced" search depth for better content extraction
        results = client.search(query, search_depth="advanced", max_results=TAVILY_MAX_RESULTS)

        for result in results.get('results', []):
            title = result.get('title', '').strip()
            url = result.get('url', '').strip()
            content = result.get('content', '').strip()

            if not title or not url:
                continue

            # Skip non-event URLs to reduce processing
            skip_domains = ['youtube.com', 'twitter.com', 'facebook.com', 'instagram.com',
                           'linkedin.com', 'reddit.com', 'news.ycombinator.com']
            if any(skip_domain in url.lower() for skip_domain in skip_domains):
                continue

            # Combine text for date parsing
            text_content = f"{title} {content}"

            dates = parse_date_candidates(text_content)
            future_dates = [d for d in dates if d >= today]

            if future_dates:
                event_date = future_dates[0]
                location = detect_location(text_content)
                event_type = detect_event_type(text_content)

                events.append({
                    'title': title,
                    'date': event_date.isoformat(),
                    'location': location,
                    'type': event_type,
                    'url': url,
                    'source': 'tavily'
                })

    except Exception as e:
        print(f"[warn] Tavily search failed for query '{query}': {e}")

    return events








def categorize_event(title: str, url: str, description: str = "") -> str:
    """Categorize event as hackathon, conference, or workshop."""
    text = f"{title} {url} {description}".lower()

    # Keywords for categorization (in order of priority)
    hackathon_keywords = [
        'hackathon', 'hack', 'challenge', 'competition', 'contest', 'kaggle',
        'hackerearth', 'topcoder', 'ctf', 'capture the flag', 'coding challenge'
    ]
    conference_keywords = [
        'conference', 'conf', 'summit', 'symposium', 'convention', 'neurips',
        'icml', 'cvpr', 'aaai', 'iclr', 'ai summit', 'ml conference'
    ]
    workshop_keywords = [
        'workshop', 'tutorial', 'bootcamp', 'training', 'seminar', 'webinar',
        'course', 'academy', 'certification', 'learning path'
    ]

    # Check for strong hackathon indicators first
    if any(kw in text for kw in hackathon_keywords):
        return 'hackathons'

    # Check for conference indicators
    if any(kw in text for kw in conference_keywords):
        return 'conferences'

    # Check for workshop indicators
    if any(kw in text for kw in workshop_keywords):
        return 'workshops'

    # Default categorization based on URL patterns
    url_lower = url.lower()
    if any(domain in url_lower for domain in ['kaggle.com', 'hackerearth.com', 'topcoder.com']):
        return 'hackathons'
    elif any(domain in url_lower for domain in ['neurips.cc', 'icml.cc', 'cvpr.thecvf.com', 'aaai.org']):
        return 'conferences'
    elif any(word in text for word in ['ml summit', 'ai conference', 'deep learning conference']):
        return 'conferences'
    else:
        return 'workshops'  # Default to workshops


def collect_all_events() -> Dict[str, List[Dict[str, Any]]]:
    """Collect events from Tavily search, deduplicate globally, then categorize them."""
    print("Starting event collection...")

    # Step 1: Collect all events from all queries (no categorization yet)
    all_raw_events = []
    print("Searching for events using Tavily (token-efficient)...")
    for query in TAVILY_QUERIES:
        events = search_tavily(query)
        all_raw_events.extend(events)
        time.sleep(TAVILY_REQUEST_DELAY)  # Rate limiting

    # Step 2: Global deduplication across all events before categorization
    deduplicated_events = deduplicate_events_globally(all_raw_events)
    print(f"Collected {len(all_raw_events)} raw events, deduplicated to {len(deduplicated_events)} unique events")

    # Step 3: Categorize the deduplicated events
    categorized_events = {
        'hackathons': [],
        'conferences': [],
        'workshops': []
    }

    for event in deduplicated_events:
        category = categorize_event(event['title'], event['url'])
        categorized_events[category].append(event)

    return categorized_events




def deduplicate_events_globally(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate events across all queries using URL as primary key."""
    seen_urls = set()
    deduplicated = []

    for event in events:
        # Normalize URL for comparison (remove query params, trailing slashes, etc.)
        url = event.get('url', '').strip().lower()

        # Normalize URL: remove protocol, www, trailing slashes, query params
        import re
        from urllib.parse import urlparse
        try:
            parsed = urlparse(url)
            normalized_url = parsed.netloc + parsed.path.rstrip('/')
            # Remove common tracking parameters
            normalized_url = re.sub(r'[?&](utm_|fbclid|ref|source|campaign)=[^&]*', '', normalized_url)
        except:
            normalized_url = url

        # Use normalized URL as the definitive deduplication key
        # If we've seen this URL before, skip this event (same event, different query/date)
        if normalized_url and normalized_url not in seen_urls:
            seen_urls.add(normalized_url)
            deduplicated.append(event)

    return deduplicated


def deduplicate_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate events based on title+date."""
    seen = set()
    deduped = []

    for event in events:
        # Normalize title for comparison
        title_clean = re.sub(r'[^\w\s]', '', event['title'].lower()).strip()
        date = event['date']

        key = (title_clean, date)
        if key not in seen:
            seen.add(key)
            deduped.append(event)

    return deduped


def filter_future_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only future events and sort by date."""
    today = datetime.datetime.now(tzlocal()).date()
    future_events = []

    for event in events:
        try:
            event_date = datetime.date.fromisoformat(event['date'])
            if event_date >= today:
                future_events.append(event)
        except ValueError:
            continue  # Skip invalid dates

    # Sort by date
    future_events.sort(key=lambda x: x['date'])
    return future_events




def save_cache(events_by_category: Dict[str, List[Dict[str, Any]]]):
    """Save events to JSON cache file. Only saves real events from DuckDuckGo search."""
    # Process each category - deduplicate and filter future events only
    processed_events = {}
    for category, events in events_by_category.items():
        # Deduplicate and filter
        deduped = deduplicate_events(events)
        filtered = filter_future_events(deduped)
        processed_events[category] = filtered

    # Create cache structure with real events only (no dummy data)
    cache_data = {
        "last_updated": datetime.datetime.now(tzlocal()).isoformat(),
        "hackathons": processed_events['hackathons'],    # All real events found
        "conferences": processed_events['conferences'],  # All real events found
        "workshops": processed_events['workshops']       # All real events found
    }

    # Save to file
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, indent=2, ensure_ascii=False)

    total_real_events = sum(len(events) for events in processed_events.values())
    print(f"Cache updated with {total_real_events} real events from Tavily search")


def main():
    """Main collection function."""
    try:
        events = collect_all_events()
        save_cache(events)
        print("Event collection completed successfully")
    except Exception as e:
        print(f"Event collection failed: {e}")
        raise


if __name__ == "__main__":
    main()


# Cron schedule: run every 6 hours
# 0 */6 * * * /path/to/python /path/to/event_collector.py
