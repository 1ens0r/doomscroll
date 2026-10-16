"""Intuition: one canonical first-level-category id -> English name map (read from the raw KuaiSAR
`first_level_category_name_en` column), so figures, tables and the simulated feed all speak the same
human-readable topic names instead of raw integer ids."""
from __future__ import annotations

CAT_NAMES: dict[int, str] = {
    0: "empty", 1: "dance", 2: "music", 3: "game", 4: "beauty & makeup", 5: "style",
    6: "star entertainment", 7: "motion", 8: "appearance", 9: "funny", 10: "travel",
    11: "life", 12: "food", 13: "agriculture", 14: "education", 15: "talent", 16: "health",
    17: "animal", 18: "automobile", 19: "emotion", 20: "illustration", 21: "history",
    22: "finance", 23: "religion", 24: "astrology", 25: "parent-child", 26: "photography",
    27: "high tech", 28: "real-time info", 29: "science", 30: "statute", 31: "law",
    32: "sports", 33: "outdoors", 34: "random", 35: "military", 36: "real estate",
    37: "oddities", 38: "reading", 39: "film & short drama",
}


def cat_name(cat_id: int) -> str:
    return CAT_NAMES.get(int(cat_id), f"category {int(cat_id)}")
