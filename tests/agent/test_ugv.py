import pytest
from operation_safe_passage.agent.ugv import UGV


class TestUGV:

    @pytest.fixture
    def test_ugv(self):
        """
        Tests the UGV's __init__ method

        Returns:
            UGV: The created UGV object
        """
        test_name = "test_ugv"
        ugv = UGV(traversal_time=20, mineclear_time=60, name=test_name, current_node="(0, 0)")
        assert test_name == ugv.name
        return ugv

    def test_ugv_reset(self, test_ugv: UGV):
        """
        Tests the UGV's reset method

        Args:
            test_agent (Agent): The Agent object retured from the fixture
        """
        test_ugv.previous_node = "Test"
        test_ugv.previous_action = "Test"
        test_ugv.num_moves = 5
        test_ugv.failures.append("TEST FAILURE")
        test_ugv.mine_nodes.add("TEST NODE")
        test_ugv.reset()
        assert None == test_ugv.previous_node and None == test_ugv.previous_action and test_ugv.num_moves == 0
        assert len(test_ugv.failures) == 0
        assert len(test_ugv.mine_nodes) == 0

    @pytest.mark.parametrize(
        "node",
        [None, {"metadata": {}, "inaccessible": {"mine_presence": True}}],
        ids=["test_ugv_step_node_is_node", "test_ugv_step_normal"]
    )
    def test_step(self, test_ugv, node, mocker):
        """
        Tests the Agent's step method

        Args:
            test_agent (Agent): The Agent object retured from the fixture
            node (dict | None): The node to pass into step
            mocker (pytest.Mocker): Tool to create Mock objects
        """
        mocker.patch("operation_safe_passage.agent.ugv.Agent.move_agent")
        has_mine = test_ugv.step(5, node)
        if node == None:
            assert has_mine == False
        else:
            assert True == has_mine
            assert test_ugv.failures[0] == {"type": "mine", "node": "(0, 0)"}
            assert list(test_ugv.mine_nodes)[0] == "(0, 0)"
