import unittest
from unittest.mock import patch

from langchain_core.messages import HumanMessage

from voice_agent.llm.nodes import closing_node, taking_order_node


class AgentNodeTests(unittest.TestCase):
    def test_taking_order_uses_agent_response(self):
        state = {
            "messages": [HumanMessage(content="I want two burgers")],
            "cart": {},
            "total_price": 0.0,
            "current_node": "taking_order",
            "next_node": "taking_order",
            "discount": 0.0,
            "free_items": [],
            "last_response": "",
            "hallucination_warning": False,
            "state_timeout_ms": 8_000,
        }

        with patch(
            "voice_agent.llm.nodes.get_agent_response",
            return_value=("Agent says burgers are added.", {"burger": 2}, 0.0, []),
        ) as agent:
            updated = taking_order_node(state)

        agent.assert_called_once()
        self.assertEqual(updated["last_response"], "Agent says burgers are added.")
        self.assertEqual(updated["cart"], {"burger": 2})
        self.assertEqual(updated["total_price"], 11.98)

    def test_closing_uses_agent_response_for_cart_changes(self):
        state = {
            "messages": [HumanMessage(content="yes add that")],
            "cart": {"burger": 1},
            "total_price": 5.99,
            "current_node": "upsell",
            "next_node": "closing",
            "discount": 0.0,
            "free_items": [],
            "last_response": "",
            "hallucination_warning": False,
            "state_timeout_ms": 4_000,
        }

        with patch(
            "voice_agent.llm.nodes.get_agent_response",
            return_value=("Agent closes with the updated total.", {"burger": 1, "shake": 1}, 0.0, []),
        ) as agent:
            updated = closing_node(state)

        agent.assert_called_once()
        self.assertEqual(updated["last_response"], "Agent closes with the updated total.")
        self.assertEqual(updated["cart"], {"burger": 1, "shake": 1})
        self.assertEqual(updated["total_price"], 8.98)


if __name__ == "__main__":
    unittest.main()
