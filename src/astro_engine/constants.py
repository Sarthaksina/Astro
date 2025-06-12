"""
Centralized constants for the Cosmic Market Oracle.
Contains all planetary and astrological constants used across the application.
"""

import swisseph as swe

# Calculation flags
GEOCENTRIC = 0  # Default calculation from Earth's perspective
HELIOCENTRIC = 1  # Calculation from Sun's perspective

# Planet IDs from Swiss Ephemeris
SUN = swe.SUN
MOON = swe.MOON
MERCURY = swe.MERCURY
VENUS = swe.VENUS
MARS = swe.MARS
JUPITER = swe.JUPITER
SATURN = swe.SATURN
URANUS = swe.URANUS
NEPTUNE = swe.NEPTUNE
PLUTO = swe.PLUTO
RAHU = swe.MEAN_NODE  # North Node
KETU = -1  # South Node (calculated from Rahu)

# Planet names mapping
PLANET_NAMES = {
    SUN: "Sun",
    MOON: "Moon",
    MERCURY: "Mercury",
    VENUS: "Venus",
    MARS: "Mars",
    JUPITER: "Jupiter",
    SATURN: "Saturn",
    URANUS: "Uranus",
    NEPTUNE: "Neptune",
    PLUTO: "Pluto",
    RAHU: "Rahu",
    KETU: "Ketu"
}

def get_planet_name(planet_id: int) -> str:
    """
    Get the name of a planet from its Swiss Ephemeris ID.
    
    Args:
        planet_id: Swiss Ephemeris planet ID
        
    Returns:
        Name of the planet
    """
    return PLANET_NAMES.get(planet_id, f"Unknown Planet {planet_id}")

# Financial significance of planets (primarily for Dasha systems and interpretations)
# This mapping provides insights into how each planet's period might influence market sectors or trends.
PLANET_FINANCIAL_SIGNIFICANCE = {
    SUN: {
        "trend": "neutral",
        "volatility": "moderate",
        "sectors": ["government", "gold", "energy", "leadership"],
        "description": "Represents authority, government policy, and leadership"
    },
    MOON: {
        "trend": "variable",
        "volatility": "high",
        "sectors": ["public sentiment", "real estate", "food", "retail"],
        "description": "Represents public sentiment, liquidity, and consumer behavior"
    },
    MERCURY: {
        "trend": "neutral",
        "volatility": "high",
        "sectors": ["communication", "technology", "media", "trade"],
        "description": "Represents communication, technology, and trading activity"
    },
    VENUS: {
        "trend": "bullish",
        "volatility": "low",
        "sectors": ["luxury", "entertainment", "fashion", "art"],
        "description": "Represents luxury goods, entertainment, and growth sectors"
    },
    MARS: {
        "trend": "bearish",
        "volatility": "very high",
        "sectors": ["military", "construction", "sports", "manufacturing"],
        "description": "Represents aggressive action, competition, and conflict"
    },
    JUPITER: {
        "trend": "bullish",
        "volatility": "low",
        "sectors": ["finance", "education", "legal", "religion"],
        "description": "Represents expansion, optimism, and growth"
    },
    SATURN: {
        "trend": "bearish",
        "volatility": "low",
        "sectors": ["infrastructure", "mining", "agriculture", "old industries"],
        "description": "Represents contraction, discipline, and long-term trends"
    },
    RAHU: { # North Node
        "trend": "variable",
        "volatility": "extreme",
        "sectors": ["foreign markets", "innovation", "speculation", "new technologies"],
        "description": "Represents speculation, innovation, and sudden growth"
    },
    KETU: { # South Node
        "trend": "bearish",
        "volatility": "extreme",
        "sectors": ["spiritual", "pharmaceuticals", "hidden sectors", "dissolution"],
        "description": "Represents dissolution, spiritual values, and hidden factors"
    }
}

# Zodiac signs (0-indexed: Aries=0, Taurus=1, ..., Pisces=11)
ARIES = 0
TAURUS = 1
GEMINI = 2
CANCER = 3
LEO = 4
VIRGO = 5
LIBRA = 6
SCORPIO = 7
SAGITTARIUS = 8
CAPRICORN = 9
AQUARIUS = 10
PISCES = 11

ZODIAC_SIGN_NAMES = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]

# Sign rulers mapping (Sign Index -> Planet ID)
SIGN_RULERS = {
    ARIES: MARS,
    TAURUS: VENUS,
    GEMINI: MERCURY,
    CANCER: MOON,
    LEO: SUN,
    VIRGO: MERCURY,
    LIBRA: VENUS,
    SCORPIO: MARS,
    SAGITTARIUS: JUPITER,
    CAPRICORN: SATURN,
    AQUARIUS: SATURN, # Traditional ruler
    PISCES: JUPITER  # Traditional ruler
}

def get_sign_ruler(sign_index: int) -> int:
    """
    Get the ruling planet of a zodiac sign.

    Args:
        sign_index: Index of the sign (0 for Aries, 1 for Taurus, etc.)

    Returns:
        Planet ID of the ruling planet. Defaults to SUN if sign_index is invalid.
    """
    return SIGN_RULERS.get(sign_index, SUN)

# Vimshottari Dasha related constants
VIMSHOTTARI_PERIODS = {
    KETU: 7,
    VENUS: 20,
    SUN: 6,
    MOON: 10,
    MARS: 7,
    RAHU: 18,
    JUPITER: 16,
    SATURN: 19,
    MERCURY: 17
}

VIMSHOTTARI_ORDER = [
    KETU, VENUS, SUN, MOON, MARS, RAHU, JUPITER, SATURN, MERCURY
]

# Nakshatra Properties
NAKSHATRA_PROPERTIES = {
    1: {"name": "Ashwini", "financial_nature": "Volatile", "ruler": KETU},
    2: {"name": "Bharani", "financial_nature": "Bearish", "ruler": VENUS}, # Adjusted to Bearish from Bearish/Volatile
    3: {"name": "Krittika", "financial_nature": "Volatile", "ruler": SUN},
    4: {"name": "Rohini", "financial_nature": "Bullish", "ruler": MOON},
    5: {"name": "Mrigashira", "financial_nature": "Volatile", "ruler": MARS}, # Adjusted to Volatile from Neutral/Volatile
    6: {"name": "Ardra", "financial_nature": "Bearish", "ruler": RAHU}, # Adjusted to Bearish from Bearish/Volatile
    7: {"name": "Punarvasu", "financial_nature": "Bullish", "ruler": JUPITER},
    8: {"name": "Pushya", "financial_nature": "Bullish", "ruler": SATURN},
    9: {"name": "Ashlesha", "financial_nature": "Bearish", "ruler": MERCURY}, # Adjusted to Bearish from Bearish/Volatile
    10: {"name": "Magha", "financial_nature": "Volatile", "ruler": KETU}, # Adjusted to Volatile from Neutral/Volatile
    11: {"name": "Purva Phalguni", "financial_nature": "Bullish", "ruler": VENUS},
    12: {"name": "Uttara Phalguni", "financial_nature": "Bullish", "ruler": SUN},
    13: {"name": "Hasta", "financial_nature": "Bullish", "ruler": MOON}, # Adjusted to Bullish from Bullish/Neutral
    14: {"name": "Chitra", "financial_nature": "Volatile", "ruler": MARS},
    15: {"name": "Swati", "financial_nature": "Volatile", "ruler": RAHU}, # Adjusted to Volatile from Neutral/Volatile
    16: {"name": "Vishakha", "financial_nature": "Volatile", "ruler": JUPITER}, # Adjusted to Volatile from Bullish/Volatile
    17: {"name": "Anuradha", "financial_nature": "Bullish", "ruler": SATURN},
    18: {"name": "Jyeshtha", "financial_nature": "Bearish", "ruler": MERCURY}, # Adjusted to Bearish from Bearish/Volatile
    19: {"name": "Mula", "financial_nature": "Bearish", "ruler": KETU}, # Adjusted to Bearish from Bearish/Volatile
    20: {"name": "Purva Ashadha", "financial_nature": "Bullish", "ruler": VENUS},
    21: {"name": "Uttara Ashadha", "financial_nature": "Bullish", "ruler": SUN},
    22: {"name": "Shravana", "financial_nature": "Bullish", "ruler": MOON}, # Adjusted to Bullish from Bullish/Neutral
    23: {"name": "Dhanishta", "financial_nature": "Volatile", "ruler": MARS}, # Adjusted to Volatile from Bullish/Volatile
    24: {"name": "Shatabhisha", "financial_nature": "Volatile", "ruler": RAHU}, # Adjusted to Volatile from Neutral/Volatile
    25: {"name": "Purva Bhadrapada", "financial_nature": "Bearish", "ruler": JUPITER}, # Adjusted to Bearish from Bearish/Volatile
    26: {"name": "Uttara Bhadrapada", "financial_nature": "Bullish", "ruler": SATURN}, # Adjusted to Bullish from Bullish/Neutral
    27: {"name": "Revati", "financial_nature": "Bullish", "ruler": MERCURY}
}

# Sector affinities for planets (used in portfolio_construction.AstrologicalSectorRotation)
PLANET_SECTOR_AFFINITIES = {
    SUN: ["Technology", "Energy", "Gold"],
    MOON: ["Real Estate", "Consumer Staples", "Utilities"],
    MERCURY: ["Technology", "Communication", "Financial Services"],
    VENUS: ["Luxury Goods", "Entertainment", "Financial Services"],
    MARS: ["Energy", "Industrial", "Defense"],
    JUPITER: ["Financial Services", "Education", "International"],
    SATURN: ["Real Estate", "Utilities", "Basic Materials"],
    RAHU: ["Technology", "Pharmaceuticals", "Speculative"], # Corresponds to North Node
    KETU: ["Pharmaceuticals", "Alternative Energy", "Precious Metals"] # Corresponds to South Node
}

# Sector affinities for Zodiac signs (0 for Aries, 1 for Taurus, ..., 11 for Pisces)
# (used in portfolio_construction.AstrologicalSectorRotation)
SIGN_SECTOR_AFFINITIES = {
    ARIES: ["Energy", "Defense", "Sports"], # 0
    TAURUS: ["Banking", "Real Estate", "Luxury Goods"], # 1
    GEMINI: ["Media", "Technology", "Transportation"], # 2
    CANCER: ["Food", "Real Estate", "Shipping"], # 3
    LEO: ["Entertainment", "Gold", "Luxury Goods"], # 4
    VIRGO: ["Healthcare", "Food Processing", "Utilities"], # 5
    LIBRA: ["Luxury Goods", "Art", "Legal Services"], # 6
    SCORPIO: ["Financial Services", "Research", "Defense"], # 7
    SAGITTARIUS: ["Travel", "Education", "International"], # 8
    CAPRICORN: ["Infrastructure", "Energy", "Government"], # 9
    AQUARIUS: ["Technology", "Aerospace", "Alternative Energy"], # 10
    PISCES: ["Pharmaceuticals", "Entertainment", "Oil & Gas"] # 11
}
