import pytest
from unittest.mock import patch
from operation_safe_passage.agent.agent import Agent


class TestAgent:

    @pytest.fixture
    def test_agent(self):
        """
        Tests the Agent's __init__ method

        Returns:
            Agent: The created Agent object
        """
        agent = Agent(type="test_agent", current_node="(0, 0)")
        assert agent.current_node == "(0, 0)"
        return agent

    def test_agent_reset(self, test_agent: Agent):
        """
        Tests the Agent's reset method

        Args:
            test_agent (Agent): The Agent object retured from the fixture
        """
        test_agent.previous_node = "Test"
        test_agent.previous_action = "Test"
        test_agent.num_moves = 5
        test_agent.reset()
        assert None == test_agent.previous_node and None == test_agent.previous_action and test_agent.num_moves == 0

    @patch.object(Agent, "move_agent")
    def test_agent_step(self, mock_move_agent, test_agent: Agent):
        """
        Tests the Agent's step method

        Args:
            mock_move_agent (MagicMock): A mocked version of the move_agent function
            test_agent (Agent): The Agent object retured from the fixture
        """
        test_agent.step(5, "(0, 0)")
        assert 5 == test_agent.previous_action
        mock_move_agent.assert_called_with("(0, 0)")

    @pytest.mark.parametrize(
        "input",
        ["(0, 1)", {"id": "(0, 1)"}],
        ids=["test_string_id", "test_id_in_dict"]
    )
    def test_move_agent(self, test_agent: Agent, input: str | dict):
        """
        Tests the Agent's move_agent method

        Args:
            test_agent (Agent): The Agent object retured from the fixture
            input (str | dict): The input into move_agent
        """
        test_agent.move_agent(input)
        assert "(0, 0)" == test_agent.previous_node
        assert "(0, 1)" == test_agent.current_node
        assert ['(0, 0)', '(0, 1)'] == test_agent.traversed_nodes
        assert 1 == test_agent.num_moves
