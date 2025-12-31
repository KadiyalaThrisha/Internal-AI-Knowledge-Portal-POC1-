#!/usr/bin/env python3
"""
Hackathons Module

Provides get_events() function that reads hackathon events from the cached JSON file.
No network calls - reads only from local cache.
"""

import json
import os
from typing import List, Dict, Any

CACHE_FILE = "events_cache.json"

def get_events() -> List[Dict[str, Any]]:
    """
    Get hackathon events from cached data.

    Returns:
        List of hackathon events, each with keys: title, date, location, url, source
    """
    try:
        if not os.path.exists(CACHE_FILE):
            print(f"[warn] Cache file {CACHE_FILE} not found")
            return []

        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)

        hackathons = cache_data.get('hackathons', [])
        return hackathons

    except Exception as e:
        print(f"[error] Failed to read hackathon events from cache: {e}")
        return []
