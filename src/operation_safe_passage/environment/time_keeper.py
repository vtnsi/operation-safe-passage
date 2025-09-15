class TimeKeeper:
    def __init__(self):
        """
        task_durations: dict {task_name: duration}
        """
        self.current_time = 0.0
        self.pending_tasks = []

    def get_current_time(self):
        """
        Gets the current time

        Returns:
            float: The current time
        """
        return self.current_time

    def add_time(self, seconds: float) -> float:
        """
        Advance time by an arbitrary number of seconds (e.g., per-scanner scan time).
        """
        self.current_time += float(seconds)
        return self.current_time
