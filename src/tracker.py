from shapely.geometry import LineString

class TrafficTracker:
    def __init__(self, blue_line_coords):
        self.blue_line = LineString(blue_line_coords)
        self.yellow_lines = []
        self.track_history = {}
        self.crossed_ids = set()
        self.violation_history = set()
        self.total_crossed = 0
        self.total_violations = 0

    def update_blue_line(self, coords):
        self.blue_line = LineString(coords)

    def add_yellow_line(self, coords):
        self.yellow_lines.append(LineString(coords))

    def clear_yellow_lines(self):
        self.yellow_lines = []
        self.violation_history = set()
        self.total_violations = 0

    def process_movement(self, track_id, prev_pos, curr_pos):
        movement = LineString([prev_pos, curr_pos])
        event_status = {"crossed": False, "violation": False}

        if movement.intersects(self.blue_line) and track_id not in self.crossed_ids:
            self.total_crossed += 1
            self.crossed_ids.add(track_id)
            event_status["crossed"] = True

        for i, lane_line in enumerate(self.yellow_lines):
            if movement.intersects(lane_line) and (track_id, i) not in self.violation_history:
                self.total_violations += 1
                self.violation_history.add((track_id, i))
                event_status["violation"] = True

        return event_status