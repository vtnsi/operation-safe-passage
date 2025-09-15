import pytest
from operation_safe_passage.environment.time_keeper import TimeKeeper


class TestTimeKeeper:

    @pytest.fixture
    def test_time_keeper(self):
        keeper = TimeKeeper()
        assert keeper.current_time == 0.0
        assert len(keeper.pending_tasks) == 0
        return keeper

    def test_get_current_time(self, test_time_keeper: TimeKeeper):
        """
        Function to test the TimeKeeper's get_current_time method

        Args:
            test_time_keeper: The created time keeper passed from the fixture
        """
        assert test_time_keeper.get_current_time() == 0.0

    def test_add_time(self, test_time_keeper: TimeKeeper):
        """
        Function to test the Environments's add_time method

        Args:
            test_time_keeper: The created TimeKeeper passed from the fixture
        """
        assert 5 == test_time_keeper.add_time(5)
