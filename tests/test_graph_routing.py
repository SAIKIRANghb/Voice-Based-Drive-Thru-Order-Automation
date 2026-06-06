import unittest
from unittest.mock import patch

from langchain_core.messages import HumanMessage

from voice_agent.llm.graph import route_turn


def state_for(next_node: str, text: str):
    return {
        "messages": [HumanMessage(content=text)],
        "cart": {"burger": 1},
        "total_price": 5.99,
        "current_node": next_node,
        "next_node": next_node,
        "discount": 0.0,
        "free_items": [],
        "last_response": "Please confirm your order.",
        "hallucination_warning": False,
        "state_timeout_ms": 6_000,
    }


class GraphRoutingTests(unittest.TestCase):
    def test_taking_order_uses_semantic_next_node(self):
        with patch("voice_agent.llm.graph.semantic_next_node", return_value="confirming"):
            self.assertEqual(route_turn(state_for("taking_order", "I am ready to pay")), "confirming")

    def test_confirming_uses_semantic_next_node(self):
        with patch("voice_agent.llm.graph.semantic_next_node", return_value="upsell"):
            self.assertEqual(route_turn(state_for("confirming", "that sounds good")), "upsell")

    def test_late_order_change_uses_semantic_next_node(self):
        with patch("voice_agent.llm.graph.semantic_next_node", return_value="taking_order"):
            self.assertEqual(route_turn(state_for("upsell", "actually swap the drink")), "taking_order")
            self.assertEqual(route_turn(state_for("closing", "remove the fries")), "taking_order")


if __name__ == "__main__":
    unittest.main()
