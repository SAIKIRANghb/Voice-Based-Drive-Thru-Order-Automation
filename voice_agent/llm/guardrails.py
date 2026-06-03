import re
from voice_agent.config import CLARIFY_TEMPLATE, FOOD_NOUNS, MENU_WHITELIST

class HallucinationGuard:
    def __init__(self, whitelist=MENU_WHITELIST, food_nouns=FOOD_NOUNS):
        self.whitelist = whitelist
        self.food_nouns = food_nouns
        
    def validate_response(self, text: str) -> tuple[str, bool]:
        """
        Validate generated text against the menu whitelist before TTS synthesis.
        If the model mentions an off-menu food item, intercept with clarification.
        
        Returns:
            (validated_text, was_intercepted)
        """
        clean_text = re.sub(r"[^\w\s]", " ", text.lower())
        clean_text = re.sub(r"\s+", " ", clean_text).strip()

        hallucinated_detected = []
        for food in sorted(self.food_nouns, key=len, reverse=True):
            if food in self.whitelist:
                continue
            if re.search(rf"\b{re.escape(food)}\b", clean_text):
                hallucinated_detected.append(food)
        
        if hallucinated_detected:
            suggestions = "burgers or nuggets" if len(hallucinated_detected) == 1 else "burgers or fries"
            clarify_text = CLARIFY_TEMPLATE.format(suggestions=suggestions)
            return (
                f"{clarify_text} We do not serve {', '.join(hallucinated_detected[:2])}.",
                True,
            )
            
        return text, False
