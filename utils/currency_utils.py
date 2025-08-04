import regex as re

# Regex pattern for detecting currency amounts in various symbols and words
MONEY_PATTERN = re.compile(
    r"(?:(?:\p{Sc})\s?[\d,.]+(?:\.\d{1,2})?)|"  # currency symbols + amount
    r"[\d,.]+\s?(usd|eur|gbp|inr|aud|cad|₽|₹|€|£|\$)",  # amount + currency words/symbols
    re.IGNORECASE
)

# Also check column headers keywords that hint money columns
MONEY_KEYWORDS = [
    "amount",
    "budget",
    "price",
    "cost",
    "total",
    "value",
    "usd",
    "inr",
    "eur",
    "gbp",
    "aud",
    "cad",
    "₹",
    "$",
    "€",
    "£"
]

def contains_money(df):
    # Normalize column names
    cols = [str(c).lower() for c in df.columns]

    # Check if any col header contains keywords
    if any(any(keyword in col for keyword in MONEY_KEYWORDS) for col in cols):
        return True

    # Check cell contents for money pattern
    for col in df.columns:
        for cell in df[col].astype(str):
            if MONEY_PATTERN.search(cell):
                return True
    return False
