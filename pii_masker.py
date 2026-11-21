import re
import json

class PIIMasker:
    """
    A utility class to perform Context-Preserving De-identification of sensitive fields.
    It replaces names, ages, emails, and phone numbers with bracketed realistic surrogates
    (e.g., [Vikram], [31_years_old], [user1@mockmail.com]) to preserve grammatical context
    for the LLM while protecting privacy.
    """
    
    # Common English and Gujarati-transliterated stop words to avoid masking as names
    STOP_WORDS = {
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", 
        "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", 
        "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", 
        "theirs", "themselves", "what", "which", "who", "whom", "this", "that", 
        "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", 
        "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", 
        "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", 
        "at", "by", "for", "with", "about", "against", "between", "into", "through", 
        "during", "before", "after", "above", "below", "to", "from", "up", "down", 
        "in", "out", "on", "off", "over", "under", "again", "further", "then", 
        "once", "here", "there", "when", "where", "why", "how", "all", "any", 
        "both", "each", "few", "more", "most", "other", "some", "such", "no", 
        "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", 
        "t", "can", "will", "just", "don", "should", "now", "d", "ll", "m", "o", 
        "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn", "hasn", 
        "haven", "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn", 
        "wasn", "weren", "won", "wouldn",
        # GujLish/Gujarati common transliterations
        "pan", "ne", "che", "na", "nu", "ni", "no", "mathi", "ane", "kem", "su", 
        "pela", "peli", "pelu", "hu", "ame", "tame", "te", "la", "lo", "le", "ja", 
        "maru", "tamru", "e", "aa", "o", "pn", "chhe"
    }

    # Lists of mock surrogates for name masking
    MOCK_FIRST_NAMES = [
        "Vikram", "Sanjay", "Neha", "Rahul", "Priya", "Amit", "Anjali", "Vijay",
        "Karan", "Simran", "Arjun", "Aditi", "Rohan", "Sneha", "Kabir", "Meera",
        "David", "Emma", "Sarah", "John", "Michael", "Lisa", "James", "Karen"
    ]
    MOCK_LAST_NAMES = [
        "Sharma", "Verma", "Gupta", "Mehta", "Shah", "Singh", "Das", "Joshi",
        "Iyer", "Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis"
    ]

    def __init__(self):
        # Compiled patterns for masking
        self.email_pattern = re.compile(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b')
        self.phone_pattern = re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,5}\)?[-.\s]?\d{3,5}[-.\s]?\d{4,5}\b')
        self.age_pattern = re.compile(r'\b\d{1,2}\s*(?:years?\s*old|yo|years? of age)\b|\bage\s*\d{1,2}\b|\b\d{1,2}-year-old\b', re.IGNORECASE)
        self.gender_pattern = re.compile(r'\b(?:male|female|gender:\s*(?:male|female))\b', re.IGNORECASE)
        
        # Name detection heuristic: capitalized words not starting sentences or preceded by name-trigger words
        self.prefix_triggers = re.compile(r'\b(?:met|spoke to|called|with|friend|boss|mentor|partner|alias|dr\.|mr\.|ms\.|mrs\.|brother|sister|father|mother|cousin|uncle|aunt|colleague|coworker|by|name|named|is)\s+([A-Z][a-zA-Z]+)\b')

    def mask_text(self, text: str) -> tuple[str, dict[str, str]]:
        if not text:
            return text, {}

        mapping = {}
        # Keep track of assigned surrogates to remain consistent
        surrogate_cache = {}
        counter = {"NAME_FIRST": 0, "NAME_LAST": 0, "EMAIL": 1, "PHONE": 1}

        def get_surrogate_name(val, is_last=False):
            key = (val.lower(), is_last)
            if key in surrogate_cache:
                return surrogate_cache[key]
            
            if is_last:
                surrogate = self.MOCK_LAST_NAMES[counter["NAME_LAST"] % len(self.MOCK_LAST_NAMES)]
                counter["NAME_LAST"] += 1
            else:
                surrogate = self.MOCK_FIRST_NAMES[counter["NAME_FIRST"] % len(self.MOCK_FIRST_NAMES)]
                counter["NAME_FIRST"] += 1
            
            bracketed = f"[{surrogate}]"
            surrogate_cache[key] = bracketed
            mapping[bracketed] = val
            return bracketed

        def shift_age(age_str):
            # Extract digits, add 3 years to shift age
            digits = re.findall(r'\d+', age_str)
            if not digits:
                return f"[{age_str}]"
            
            original_val = digits[0]
            shifted_val = str(int(original_val) + 3)
            # Reconstruct the string with shifted age
            shifted_str = age_str.replace(original_val, shifted_val)
            bracketed = f"[{shifted_str.replace(' ', '_')}]"
            mapping[bracketed] = age_str
            return bracketed

        # 1. Mask Email
        def email_repl(match):
            val = match.group(0)
            placeholder = f"[user{counter['EMAIL']}@mockmail.com]"
            counter["EMAIL"] += 1
            mapping[placeholder] = val
            return placeholder
        masked_text = self.email_pattern.sub(email_repl, text)

        # 2. Mask Phone
        def phone_repl(match):
            val = match.group(0)
            placeholder = f"[+91-90000-1000{counter['PHONE']}]"
            counter["PHONE"] += 1
            mapping[placeholder] = val
            return placeholder
        masked_text = self.phone_pattern.sub(phone_repl, masked_text)

        # 3. Mask Age with shifting
        def age_repl(match):
            return shift_age(match.group(0))
        masked_text = self.age_pattern.sub(age_repl, masked_text)

        # 4. Mask Gender (bracketed for de-identification, but preserve semantics)
        def gender_repl(match):
            val = match.group(0)
            placeholder = f"[{val.lower()}]"
            mapping[placeholder] = val
            return placeholder
        masked_text = self.gender_pattern.sub(gender_repl, masked_text)

        # 5. Mask Names using trigger prefixes
        def prefix_repl(match):
            full_match = match.group(0)
            name_part = match.group(1)
            if name_part.lower() in self.STOP_WORDS:
                return full_match
            # We assume first names here for simple triggers
            placeholder = get_surrogate_name(name_part, is_last=False)
            return full_match.replace(name_part, placeholder)
        
        masked_text = self.prefix_triggers.sub(prefix_repl, masked_text)

        # 6. Mask any remaining standalone capitalized words that might be names (excluding start of sentences)
        # We split text into sentences to avoid masking first word
        sentences = re.split(r'([.!?]\s+)', masked_text)
        for idx in range(len(sentences)):
            # Only process sentence content, not delimiters
            if idx % 2 == 0:
                words = sentences[idx].split(' ')
                for word_idx in range(1, len(words)):  # Skip the very first word in the sentence segment
                    word = words[word_idx]
                    # Skip already masked placeholders (like [Emma] or [31_years_old])
                    if '[' in word or ']' in word:
                        continue
                    clean_word = re.sub(r'[^\w]', '', word)
                    if clean_word and clean_word[0].isupper() and clean_word.lower() not in self.STOP_WORDS:
                        # Simple heuristic: if the previous word was also masked as a name, this is likely a surname
                        is_last = False
                        if word_idx > 1:
                            prev_word = words[word_idx - 1]
                            if prev_word.startswith('[') and prev_word.endswith(']'):
                                # Check if it's in our first name mock list to guess if it's a name placeholder
                                content_inside = prev_word[1:-1]
                                if content_inside in self.MOCK_FIRST_NAMES:
                                    is_last = True
                        
                        placeholder = get_surrogate_name(clean_word, is_last=is_last)
                        words[word_idx] = word.replace(clean_word, placeholder)
                sentences[idx] = ' '.join(words)
        
        masked_text = ''.join(sentences)

        return masked_text, mapping

    def unmask_text(self, text: str, mapping: dict[str, str]) -> str:
        if not text or not mapping:
            return text

        unmasked_text = text
        # Sort keys descending by length to avoid partial replacements (e.g. [31_years_old] vs [31])
        for placeholder in sorted(mapping.keys(), key=len, reverse=True):
            original_value = mapping[placeholder]
            unmasked_text = unmasked_text.replace(placeholder, original_value)
            # Support replaced quotes/escapes or stripped brackets
            unmasked_text = unmasked_text.replace(placeholder.replace("[", "").replace("]", ""), original_value)
        return unmasked_text

    def mask_messages(self, messages: list[dict]) -> tuple[list[dict], dict[str, str]]:
        """Mask a list of message dicts for OpenAI format, returning the masked list and mapping."""
        masked_messages = []
        global_mapping = {}

        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                masked_content, mapping = self.mask_text(content)
                global_mapping.update(mapping)
                masked_messages.append({**msg, "content": masked_content})
            elif isinstance(content, list):
                # Handle multimodal (text + image url) message content
                new_content = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        masked_text, mapping = self.mask_text(item.get("text", ""))
                        global_mapping.update(mapping)
                        new_content.append({**item, "text": masked_text})
                    else:
                        new_content.append(item)
                masked_messages.append({**msg, "content": new_content})
            else:
                masked_messages.append(msg)

        return masked_messages, global_mapping
