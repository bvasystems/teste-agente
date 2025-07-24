from agentle.vector_stores.filters.match_any import MatchAny
from agentle.vector_stores.filters.match_except import MatchExcept
from agentle.vector_stores.filters.match_phrase import MatchPhrase
from agentle.vector_stores.filters.match_text import MatchText
from agentle.vector_stores.filters.match_value import MatchValue


Match = MatchValue | MatchText | MatchPhrase | MatchAny | MatchExcept
