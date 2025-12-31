#!/usr/bin/env python3
"""
Conferences Module

Provides get_events() function that reads conference events from the cached JSON file.
No network calls - reads only from local cache.
"""

import json
import os
from typing import List, Dict, Any

CACHE_FILE = "events_cache.json"

def get_events() -> List[Dict[str, Any]]:
    """
    Get conference events from cached data.

    Returns:
        List of conference events, each with keys: title, date, location, url, source
    """
    try:
        if not os.path.exists(CACHE_FILE):
            print(f"[warn] Cache file {CACHE_FILE} not found")
            return []

        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)

        conferences = cache_data.get('conferences', [])
        return conferences

    except Exception as e:
        print(f"[error] Failed to read conference events from cache: {e}")
        return []
