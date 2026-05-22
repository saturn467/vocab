# ---------------------------------------------------------------------------
# Vocabulary Taxonomer – Configuration
# ---------------------------------------------------------------------------

# ── Paths ─────────────────────────────────────────────────────────────────
import pathlib as _pathlib

_PROJECT_ROOT = _pathlib.Path(__file__).resolve().parent.parent

# ── Excel source ──────────────────────────────────────────────────────────
EXCEL_FILE = str(_PROJECT_ROOT / "data" / "words.xlsx")
LEMMA_COLUMN = "lemme"
FREQUENCY_COLUMN = "freq"
CONTEXT_COLUMN = "Context"
CONTEXT_PLACEHOLDER = "Reverso"  # value meaning "no real context"

# ── Language pair ─────────────────────────────────────────────────────────
SOURCE_LANGUAGE = "en"
TARGET_LANGUAGE = "fr"

# ── Embedding model ───────────────────────────────────────────────────────
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# ── CEFR frequency-rank bands ────────────────────────────────────────────
CEFR_BANDS: list[tuple[str, int, int]] = [
    ("A1",    1,   500),
    ("A2",  501,  1500),
    ("B1", 1501,  3500),
    ("B2", 3501,  6000),
    ("C1", 6001,  8500),
    ("C2", 8501, 99999),
]

# ── Deck sizing ───────────────────────────────────────────────────────────
TARGET_DECK_SIZE = 15
MIN_DECK_SIZE = 10
MAX_DECK_SIZE = 20

# ── Quality gate ──────────────────────────────────────────────────────────
MIN_QUALITY_SCORE = 0.30          # decks below this Q are rejected
OUTLIER_SIMILARITY_THRESHOLD = 0.15  # words below this sim-to-centroid ejected
MIN_DECKS = 100                   # hard floor on output deck count

# ── Theme taxonomy (theme → subcategories used for deck naming) ───────────
THEMES: dict[str, list[str]] = {
    "Food & Dining": [
        "Ingredients & Produce",
        "Cooking & Kitchen",
        "Dining & Table Setting",
        "Beverages & Drinks",
        "Restaurants & Eating Out",
        "Nutrition & Diet",
        "Baking & Desserts",
    ],
    "Animals & Nature": [
        "Domestic Pets",
        "Wild Animals",
        "Birds & Insects",
        "Plants & Trees",
        "Marine Life",
        "Habitats & Ecosystems",
        "Farming & Agriculture",
    ],
    "Home & Daily Life": [
        "Furniture & Household",
        "Cleaning & Chores",
        "Rooms & Spaces",
        "Routine & Habits",
        "Tools & Repairs",
        "Gardening & Outdoors",
        "Appliances & Utilities",
    ],
    "Travel & Transportation": [
        "Vehicles & Driving",
        "Public Transport",
        "Airports & Flying",
        "Navigation & Directions",
        "Hotels & Accommodation",
        "Tourism & Sightseeing",
        "Luggage & Packing",
    ],
    "Health & Body": [
        "Body Parts & Anatomy",
        "Illness & Symptoms",
        "Medicine & Treatment",
        "Hospital & Doctors",
        "Fitness & Exercise",
        "Mental Health & Wellbeing",
        "Hygiene & Self-care",
    ],
    "Work & Business": [
        "Office & Workplace",
        "Jobs & Professions",
        "Meetings & Communication",
        "Finance & Banking",
        "Trade & Commerce",
        "Management & Leadership",
        "Employment & Hiring",
    ],
    "Education & Learning": [
        "School & Classroom",
        "Subjects & Curriculum",
        "Exams & Grades",
        "University & Research",
        "Reading & Writing",
        "Student Life",
        "Teaching & Instruction",
    ],
    "Technology & Media": [
        "Computers & Software",
        "Internet & Social Media",
        "Phones & Devices",
        "Television & Radio",
        "Photography & Video",
        "Digital Communication",
        "Cybersecurity & Privacy",
    ],
    "Arts & Entertainment": [
        "Music & Instruments",
        "Painting & Drawing",
        "Theatre & Performance",
        "Cinema & Film",
        "Literature & Books",
        "Dance & Movement",
        "Museums & Galleries",
    ],
    "Sports & Leisure": [
        "Ball Sports",
        "Water Sports",
        "Outdoor Activities",
        "Games & Hobbies",
        "Competition & Tournaments",
        "Fitness & Training",
        "Winter Sports",
    ],
    "Family & Relationships": [
        "Family Members",
        "Marriage & Partnership",
        "Friendship & Social Life",
        "Childhood & Parenting",
        "Elderly & Aging",
        "Community & Neighbours",
        "Celebrations & Events",
    ],
    "Shopping & Clothing": [
        "Clothing & Outfits",
        "Shoes & Accessories",
        "Stores & Markets",
        "Prices & Money",
        "Fashion & Style",
        "Fabrics & Materials",
        "Online Shopping",
    ],
    "Weather & Environment": [
        "Weather & Seasons",
        "Climate & Temperature",
        "Natural Disasters",
        "Pollution & Waste",
        "Sustainability & Recycling",
        "Geography & Landscape",
        "Energy & Resources",
    ],
    "Politics & Society": [
        "Government & Law",
        "Elections & Voting",
        "Rights & Freedoms",
        "Social Issues",
        "International Relations",
        "Military & Defence",
        "Media & Journalism",
    ],
    "Science & Research": [
        "Biology & Life Sciences",
        "Chemistry & Materials",
        "Physics & Engineering",
        "Space & Astronomy",
        "Mathematics & Statistics",
        "Laboratory & Experiments",
        "Medical Research",
    ],
    "Law & Justice": [
        "Courts & Trials",
        "Crime & Punishment",
        "Police & Investigation",
        "Legal Documents",
        "Rights & Regulations",
        "Prisons & Rehabilitation",
        "Contracts & Agreements",
    ],
    "Emotions & Personality": [
        "Feelings & Moods",
        "Character & Traits",
        "Attitudes & Opinions",
        "Conflict & Anger",
        "Love & Affection",
        "Fear & Anxiety",
        "Confidence & Self-esteem",
        "Surprise & Reactions",
    ],
    "Time & Numbers": [
        "Days & Months",
        "Clocks & Schedules",
        "Counting & Quantities",
        "Age & Duration",
        "Frequency & Repetition",
        "Past & Future",
        "Measurement & Units",
        "Adverbs of Degree & Certainty",
    ],
    "City & Infrastructure": [
        "Buildings & Architecture",
        "Streets & Roads",
        "Parks & Public Spaces",
        "Services & Utilities",
        "Urban Life",
        "Construction & Development",
        "Neighbourhoods & Districts",
    ],
    "Communication & Language": [
        "Speaking & Conversation",
        "Writing & Correspondence",
        "Grammar & Vocabulary",
        "Interjections & Exclamations",
        "Debates & Arguments",
        "Translation & Interpretation",
        "Public Speaking & Presentations",
        "Filler Words & Hesitations",
    ],
}

# Flat list of theme names (convenience)
THEME_NAMES: list[str] = list(THEMES.keys())

# Flat list of all subcategories with their parent theme
SUBCATEGORIES: list[tuple[str, str]] = [
    (sub, theme)
    for theme, subs in THEMES.items()
    for sub in subs
]
