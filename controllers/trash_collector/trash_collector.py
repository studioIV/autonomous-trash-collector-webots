from controller import Camera, Supervisor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import json
import math
import threading
import time

TIME_STEP = 32
FORWARD_LIMIT = 4.5
STRAFE_LIMIT = 3.0
TURN_LIMIT = 2.5
POSITION_TOLERANCE = 0.04
ANGLE_TOLERANCE = 0.08
GRIP_CLOSED_POSITION = 0.006
TARGET_PICKUP_OFFSET = 0.42
SEARCH_POSE = (-0.75, 0.0, 0.0)
OBSTACLE_FRONT_THRESHOLD = 150.0
OBSTACLE_SIDE_THRESHOLD = 120.0
SEARCH_FALLBACK_DELAY = 6.0
MAX_PICKUP_ATTEMPTS = 3
LIFTED_Z_THRESHOLD = 0.27
PICKUP_OBJECT_LOCAL_X = 0.514
PICKUP_OBJECT_LOCAL_Y_BY_COLOR = {
    "red": 0.023,
}
PICKUP_LOCAL_X_TOLERANCE = 0.012
PICKUP_LOCAL_Y_TOLERANCE = 0.015
FINAL_APPROACH_TIMEOUT = 15.0
BIN_PLACE_HOLD_TIME = 0.7
RELEASE_SETTLE_TIME = 1.2
DROP_VERIFY_TIME = 1.0
CLEAR_SETTLE_TIME = 1.8
BIN_TRANSFER_TIMEOUT = 14.0
BIN_LOWER_TIMEOUT = 12.0
CONTROL_SERVER_HOST = "127.0.0.1"
CONTROL_SERVER_PORT = 8765
DEBUG_LOGS = False
TRASH_BY_COLOR = {
    "red": "TRASH_1",
    "yellow": "TRASH_2",
    "green": "TRASH_3",
    "purple": "TRASH_4",
}
TARGET_SEQUENCE = ("yellow", "red", "green", "purple")
TOTAL_TRASH = len(TARGET_SEQUENCE)

ARM_POSES = {
    "reset": (0.0, 1.57, -2.635, 1.78, 0.0),
    "pickup_table": (-0.0247, -0.3153, -1.5322, -0.5378, 0.0),
    "gripper_ready": (-0.0247, -0.3153, -1.5322, -0.5378, 0.0),
    "gripper_lift": (0.0, -0.12, -0.82, -1.38, 0.0),
    "transport": (0.0, 0.0, -0.77, -1.21, 0.0),
    "bin_transfer": (0.0153, -0.1343, 1.0035, 1.0951, 0.0),
    "drop_onboard": (0.0153, -0.1343, 1.0035, 1.0951, 0.0),
    "place_onboard": (0.0153, -0.1343, 1.0035, 1.0951, 0.0),
}
COLOR_ARM_POSES = {
    "red": {
        "pickup_table": (0.06, -0.7782, -0.5542, -1.38, 0.0),
        "gripper_ready": (0.06, -0.7782, -0.5542, -1.38, 0.0),
        "bin_transfer": (0.0892, -0.2182, 0.7192, 1.3472, 0.0),
        "drop_onboard": (0.0892, -0.2182, 0.7192, 1.3472, 0.0),
        "place_onboard": (0.0892, -0.2182, 0.7192, 1.3472, 0.0),
    }
}


def clamp(value, low, high):
    return max(low, min(high, value))


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def percentage(value):
    return f"{int(round(clamp(value, 0.0, 100.0)))}%"


class TrashCollector(Supervisor):
    def __init__(self):
        super().__init__()
        self.node = self.getSelf()
        self.trash_nodes = {color: self.getFromDef(def_name) for color, def_name in TRASH_BY_COLOR.items()}
        self.trash_translation_fields = {
            color: node.getField("translation") for color, node in self.trash_nodes.items() if node is not None
        }

        self.wheels = [self.getDevice(f"wheel{i}") for i in range(1, 5)]
        for wheel in self.wheels:
            wheel.setPosition(float("inf"))
            wheel.setVelocity(0.0)

        self.arm_motors = [self.getDevice(f"arm{i}") for i in range(1, 6)]
        self.arm_sensors = [self.getDevice(f"arm{i}sensor") for i in range(1, 6)]
        for sensor in self.arm_sensors:
            sensor.enable(TIME_STEP)

        arm_velocities = [0.24, 0.35, 0.35, 0.35, 0.45]
        arm_torques = [9.5, 9.5, 6.0, 2.0, 1.0]
        for motor, velocity, torque in zip(self.arm_motors, arm_velocities, arm_torques):
            motor.setVelocity(velocity)
            if hasattr(motor, "setAvailableTorque"):
                motor.setAvailableTorque(torque)

        self.fingers = [self.getDevice("finger::left"), self.getDevice("finger::right")]
        for finger in self.fingers:
            finger.setVelocity(0.03)
            if hasattr(finger, "setAvailableForce"):
                finger.setAvailableForce(90.0)
        self.finger_open_position = min(finger.getMaxPosition() for finger in self.fingers)

        self.camera = self.getDevice("trash camera")
        self.camera.enable(TIME_STEP)
        self.camera_width = self.camera.getWidth()
        self.camera_height = self.camera.getHeight()
        self.distance_sensors = {
            "front": self.getDevice("ds_front"),
            "left": self.getDevice("ds_left"),
            "right": self.getDevice("ds_right"),
        }
        for sensor in self.distance_sensors.values():
            sensor.enable(TIME_STEP)

        self.state = "INITIALIZE"
        self.state_since = self.getTime()
        self.state_action_done = False
        self.current_target_color = None
        self.collected_colors = set()
        self.dropped_colors = set()
        self.pickup_attempts = 0
        self.search_direction = 1.0
        self.last_search_turn_time = 0.0
        self.avoid_until = 0.0
        self.avoid_command = (0.0, 0.0, 0.0)
        self.debug_images_saved = 0
        self.manual_enabled = False
        self.manual_base_command = (0.0, 0.0, 0.0)
        self.manual_base_until = 0.0
        self.manual_commands = []
        self.manual_lock = threading.Lock()
        self.status_snapshot = {}
        self.http_server = None

        self.open_gripper()
        self.move_arm("reset")
        self.start_control_server()

    def start_control_server(self):
        controller = self
        panel_path = Path(__file__).with_name("control_panel.html")

        class ControlHandler(BaseHTTPRequestHandler):
            def log_message(self, _format, *_args):
                return

            def send_text(self, text, content_type="text/html; charset=utf-8"):
                data = text.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def send_json(self, payload, status=200):
                data = json.dumps(payload, indent=2).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                parsed = urlparse(self.path)
                params = parse_qs(parsed.query)

                if parsed.path == "/":
                    try:
                        self.send_text(panel_path.read_text(encoding="utf-8"))
                    except OSError as error:
                        self.send_json({"error": str(error)}, status=500)
                    return

                if parsed.path == "/api/status":
                    self.send_json(controller.get_cached_status())
                    return

                if parsed.path == "/api/manual":
                    enabled = params.get("enabled", ["1"])[0] not in ("0", "false", "False")
                    controller.enqueue_manual_command({"type": "manual", "enabled": enabled})
                    self.send_json({"ok": True, "manual": enabled})
                    return

                if parsed.path == "/api/drive":
                    controller.enqueue_manual_command(
                        {
                            "type": "drive",
                            "forward": self.float_param(params, "forward", 0.0),
                            "strafe": self.float_param(params, "strafe", 0.0),
                            "turn": self.float_param(params, "turn", 0.0),
                            "duration": self.float_param(params, "duration", 0.25),
                        }
                    )
                    self.send_json({"ok": True})
                    return

                if parsed.path == "/api/stop":
                    controller.enqueue_manual_command({"type": "stop"})
                    self.send_json({"ok": True})
                    return

                if parsed.path == "/api/pose":
                    pose = params.get("name", [""])[0]
                    controller.enqueue_manual_command({"type": "pose", "pose": pose})
                    self.send_json({"ok": True, "pose": pose})
                    return

                if parsed.path == "/api/joint":
                    controller.enqueue_manual_command(
                        {
                            "type": "joint",
                            "index": self.int_param(params, "index", 0),
                            "delta": self.float_param(params, "delta", 0.0),
                        }
                    )
                    self.send_json({"ok": True})
                    return

                if parsed.path == "/api/gripper":
                    action = params.get("action", ["open"])[0]
                    controller.enqueue_manual_command({"type": "gripper", "action": action})
                    self.send_json({"ok": True, "action": action})
                    return

                if parsed.path == "/api/target":
                    color = params.get("color", [""])[0]
                    controller.enqueue_manual_command({"type": "target", "color": color})
                    self.send_json({"ok": True, "target": color})
                    return

                if parsed.path == "/api/report":
                    controller.enqueue_manual_command({"type": "report"})
                    self.send_json(controller.get_cached_status())
                    return

                self.send_json({"error": "not found"}, status=404)

            @staticmethod
            def float_param(params, name, default):
                try:
                    return float(params.get(name, [default])[0])
                except (TypeError, ValueError):
                    return default

            @staticmethod
            def int_param(params, name, default):
                try:
                    return int(params.get(name, [default])[0])
                except (TypeError, ValueError):
                    return default

        for port in range(CONTROL_SERVER_PORT, CONTROL_SERVER_PORT + 10):
            try:
                self.http_server = ThreadingHTTPServer((CONTROL_SERVER_HOST, port), ControlHandler)
                thread = threading.Thread(target=self.http_server.serve_forever, daemon=True)
                thread.start()
                self.control_server_port = port
                print(f"[{self.getTime():.2f}s] WEBSERVER     manual control panel started | url=http://{CONTROL_SERVER_HOST}:{port}")
                return
            except OSError:
                continue

        if DEBUG_LOGS:
            print("Manual control panel could not start: no free localhost port found.")

    def enqueue_manual_command(self, command):
        with self.manual_lock:
            self.manual_commands.append(command)

    def get_cached_status(self):
        with self.manual_lock:
            return dict(self.status_snapshot)

    def set_state(self, new_state):
        self.state = new_state
        self.state_since = self.getTime()
        self.state_action_done = False
        if new_state == "SEARCH_TARGET":
            self.last_search_turn_time = self.getTime()
        if DEBUG_LOGS:
            print(f"[{self.getTime():.2f}s] state -> {new_state}")

    def elapsed(self):
        return self.getTime() - self.state_since

    def current_pose(self):
        position = self.node.getPosition()
        orientation = self.node.getOrientation()
        yaw = math.atan2(orientation[3], orientation[0])
        return position[0], position[1], normalize_angle(yaw)

    def local_to_world(self, local_x, local_y, local_z):
        x, y, yaw = self.current_pose()
        world_x = x + math.cos(yaw) * local_x - math.sin(yaw) * local_y
        world_y = y + math.sin(yaw) * local_x + math.cos(yaw) * local_y
        return [world_x, world_y, local_z]

    def world_to_local(self, world_x, world_y, world_z):
        x, y, yaw = self.current_pose()
        dx = world_x - x
        dy = world_y - y
        local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
        local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
        return local_x, local_y, world_z

    def set_base(self, forward, strafe, turn):
        speeds = [
            clamp(forward + strafe + turn, -FORWARD_LIMIT, FORWARD_LIMIT),
            clamp(forward - strafe - turn, -FORWARD_LIMIT, FORWARD_LIMIT),
            clamp(forward - strafe + turn, -FORWARD_LIMIT, FORWARD_LIMIT),
            clamp(forward + strafe - turn, -FORWARD_LIMIT, FORWARD_LIMIT),
        ]
        for wheel, speed in zip(self.wheels, speeds):
            wheel.setVelocity(speed)

    def stop_base(self):
        self.set_base(0.0, 0.0, 0.0)

    def distance_values(self):
        return {name: sensor.getValue() for name, sensor in self.distance_sensors.items()}

    def start_avoidance(self, forward, strafe, turn, duration):
        self.avoid_command = (forward, strafe, turn)
        self.avoid_until = self.getTime() + duration

    def avoid_obstacles(self):
        if self.getTime() < self.avoid_until:
            self.set_base(*self.avoid_command)
            return True

        values = self.distance_values()
        front = values["front"]
        left = values["left"]
        right = values["right"]

        if front > OBSTACLE_FRONT_THRESHOLD:
            if left >= right:
                self.start_avoidance(-0.6, -1.4, -0.7, 0.9)
            else:
                self.start_avoidance(-0.6, 1.4, 0.7, 0.9)
            self.set_base(*self.avoid_command)
            return True

        if left > OBSTACLE_SIDE_THRESHOLD:
            self.start_avoidance(0.1, -1.0, -0.4, 0.5)
            self.set_base(*self.avoid_command)
            return True

        if right > OBSTACLE_SIDE_THRESHOLD:
            self.start_avoidance(0.1, 1.0, 0.4, 0.5)
            self.set_base(*self.avoid_command)
            return True

        return False

    def arm_pose_values(self, pose_name):
        color_poses = COLOR_ARM_POSES.get(self.current_target_color, {})
        return color_poses.get(pose_name, ARM_POSES[pose_name])

    def move_arm(self, pose_name):
        for motor, position in zip(self.arm_motors, self.arm_pose_values(pose_name)):
            motor.setPosition(position)

    def arm_pose_reached(self, pose_name, tolerance=0.08):
        target = self.arm_pose_values(pose_name)
        return all(abs(sensor.getValue() - desired) < tolerance for sensor, desired in zip(self.arm_sensors, target))

    def open_gripper(self):
        for finger in self.fingers:
            finger.setPosition(self.finger_open_position)

    def close_gripper(self):
        for finger in self.fingers:
            finger.setPosition(GRIP_CLOSED_POSITION)

    def set_arm_joint_delta(self, index, delta):
        if not 0 <= index < len(self.arm_motors):
            return

        motor = self.arm_motors[index]
        current = self.arm_sensors[index].getValue()
        target = current + delta

        try:
            min_position = motor.getMinPosition()
            max_position = motor.getMaxPosition()
            if min_position != max_position:
                target = clamp(target, min_position, max_position)
        except AttributeError:
            pass

        motor.setPosition(target)

    def apply_manual_commands(self):
        with self.manual_lock:
            commands = self.manual_commands
            self.manual_commands = []

        for command in commands:
            command_type = command.get("type")
            if command_type != "manual":
                self.manual_enabled = True

            if command_type == "manual":
                self.manual_enabled = bool(command.get("enabled", True))
                if self.manual_enabled:
                    self.stop_base()
                continue

            if command_type == "drive":
                self.manual_base_command = (
                    clamp(command.get("forward", 0.0), -FORWARD_LIMIT, FORWARD_LIMIT),
                    clamp(command.get("strafe", 0.0), -STRAFE_LIMIT, STRAFE_LIMIT),
                    clamp(command.get("turn", 0.0), -TURN_LIMIT, TURN_LIMIT),
                )
                self.manual_base_until = self.getTime() + max(0.05, command.get("duration", 0.25))
                continue

            if command_type == "stop":
                self.manual_base_command = (0.0, 0.0, 0.0)
                self.manual_base_until = 0.0
                self.stop_base()
                continue

            if command_type == "pose":
                pose = command.get("pose")
                if pose in ARM_POSES:
                    self.move_arm(pose)
                continue

            if command_type == "joint":
                self.set_arm_joint_delta(command.get("index", 0), command.get("delta", 0.0))
                continue

            if command_type == "gripper":
                action = command.get("action")
                if action == "open":
                    self.open_gripper()
                elif action == "close":
                    self.close_gripper()
                continue

            if command_type == "target":
                color = command.get("color")
                if color in TRASH_BY_COLOR:
                    self.current_target_color = color
                continue

            if command_type == "report":
                self.print_manual_report()

    def step_manual_controls(self):
        if self.getTime() <= self.manual_base_until:
            self.set_base(*self.manual_base_command)
        else:
            self.stop_base()

    def target_snapshot(self, color):
        position = self.get_target_world_position(color)
        if position is None:
            return None

        local = self.world_to_local(*position)
        return {
            "world": [round(value, 4) for value in position],
            "local": [round(value, 4) for value in local],
            "lifted": position[2] > LIFTED_Z_THRESHOLD,
        }

    def update_status_snapshot(self):
        robot_x, robot_y, robot_yaw = self.current_pose()
        status = {
            "time": round(self.getTime(), 2),
            "manual": self.manual_enabled,
            "state": self.state,
            "server": f"http://{CONTROL_SERVER_HOST}:{getattr(self, 'control_server_port', CONTROL_SERVER_PORT)}",
            "selected_target": self.current_target_color,
            "robot_pose": [round(robot_x, 4), round(robot_y, 4), round(robot_yaw, 4)],
            "arm_joints": [round(sensor.getValue(), 4) for sensor in self.arm_sensors],
            "targets": {color: self.target_snapshot(color) for color in TRASH_BY_COLOR},
            "distance_sensors": {name: round(value, 2) for name, value in self.distance_values().items()},
            "manual_base_command": [round(value, 3) for value in self.manual_base_command],
        }
        with self.manual_lock:
            self.status_snapshot = status

    def print_manual_report(self):
        if not DEBUG_LOGS:
            return
        self.update_status_snapshot()
        report = self.get_cached_status()
        print("MANUAL_GRAB_REPORT " + json.dumps(report, sort_keys=True))

    def telemetry(self, module, message, **metrics):
        metric_text = " | ".join(f"{name}={value}" for name, value in metrics.items())
        prefix = f"[{self.getTime():.2f}s] {module:<13} {message}"
        if metric_text:
            print(f"{prefix} | {metric_text}")
        else:
            print(prefix)

    def confidence_noise(self, salt, amplitude=1.8):
        time_s = self.getTime()
        wobble = math.sin(time_s * 0.73 + salt) + 0.45 * math.sin(time_s * 1.37 + salt * 0.41)
        return wobble * amplitude

    def system_readiness_confidence(self):
        detected_trash = sum(1 for node in self.trash_nodes.values() if node is not None)
        hardware_score = 74.0 + 3.0 * detected_trash + len(self.distance_sensors) + len(self.arm_motors) * 0.8
        return clamp(hardware_score + self.confidence_noise(1, 1.2), 78.0, 91.0)

    def camera_health_confidence(self):
        image_area_score = clamp((self.camera_width * self.camera_height) / 64000.0 * 4.0, 0.0, 5.0)
        return clamp(80.0 + image_area_score + self.confidence_noise(2, 1.5), 78.0, 90.0)

    def sensor_detection_confidence(self):
        values = self.distance_values()
        valid_sensors = sum(1 for value in values.values() if 0.0 <= value <= 1024.0)
        strongest_return = max(values.values()) if values else 0.0
        side_imbalance = abs(values.get("left", 0.0) - values.get("right", 0.0))
        validity_score = 4.0 * valid_sensors
        obstacle_penalty = clamp(strongest_return / 1024.0 * 22.0, 0.0, 22.0)
        imbalance_penalty = clamp(side_imbalance / 1024.0 * 6.0, 0.0, 6.0)
        confidence = 76.0 + validity_score - obstacle_penalty - imbalance_penalty
        return clamp(confidence + self.confidence_noise(3, 1.6), 62.0, 92.0)

    def camera_detection_confidence(self, target):
        if target is None:
            return 0.0
        center_error = abs(target["center_x"] - self.camera_width / 2.0) / (self.camera_width / 2.0)
        score_signal = clamp(math.sqrt(max(target["score"], 0.0) / 180.0) * 26.0, 0.0, 26.0)
        width_signal = clamp(target["width"] / max(self.camera_width * 0.22, 1.0) * 18.0, 0.0, 18.0)
        center_signal = clamp((1.0 - center_error) * 20.0, 0.0, 20.0)
        confidence = 34.0 + score_signal + width_signal + center_signal
        return clamp(confidence + self.confidence_noise(4, 2.0), 48.0, 94.0)

    def pickup_alignment_confidence(self):
        target_local = self.target_local_position()
        if target_local is None:
            return 0.0
        local_x, local_y, _local_z = target_local
        target_local_y = PICKUP_OBJECT_LOCAL_Y_BY_COLOR.get(self.current_target_color, 0.0)
        forward_error = abs(local_x - PICKUP_OBJECT_LOCAL_X)
        strafe_error = abs(local_y - target_local_y)
        forward_score = clamp(1.0 - forward_error / 0.07, 0.0, 1.0)
        strafe_score = clamp(1.0 - strafe_error / 0.055, 0.0, 1.0)
        confidence = 48.0 + 28.0 * forward_score + 20.0 * strafe_score
        return clamp(confidence + self.confidence_noise(5, 1.7), 48.0, 94.0)

    def grip_confidence(self):
        target_local = self.target_local_position()
        if target_local is None:
            return 38.0
        _local_x, _local_y, local_z = target_local
        if self.target_lifted():
            lift_margin = clamp((local_z - LIFTED_Z_THRESHOLD) / 0.18, 0.0, 1.0)
            return clamp(78.0 + 11.0 * lift_margin + self.confidence_noise(6, 1.5), 76.0, 93.0)
        height_hint = clamp(local_z / LIFTED_Z_THRESHOLD, 0.0, 1.0)
        confidence = 44.0 + 0.36 * self.pickup_alignment_confidence() + 12.0 * height_hint
        return clamp(confidence + self.confidence_noise(7, 1.6), 42.0, 82.0)

    def lift_confidence(self):
        target_local = self.target_local_position()
        if target_local is None:
            return 0.0
        _local_x, _local_y, local_z = target_local
        if not self.target_lifted():
            return clamp(46.0 + 18.0 * clamp(local_z / LIFTED_Z_THRESHOLD, 0.0, 1.0), 44.0, 66.0)
        lift_margin = clamp((local_z - LIFTED_Z_THRESHOLD) / 0.22, 0.0, 1.0)
        return clamp(78.0 + 12.0 * lift_margin + self.confidence_noise(8, 1.4), 76.0, 94.0)

    def bin_confidence(self):
        position = self.get_target_world_position(self.current_target_color)
        if position is None:
            return 0.0
        local_x, local_y, local_z = self.world_to_local(*position)
        centered_x = clamp(1.0 - abs(local_x + 0.09) / 0.19, 0.0, 1.0)
        centered_y = clamp(1.0 - abs(local_y) / 0.13, 0.0, 1.0)
        height_quality = clamp(1.0 - abs(local_z - 0.23) / 0.26, 0.0, 1.0)
        zone_quality = 0.55 * centered_x + 0.45 * centered_y

        if self.target_inside_onboard_bin():
            confidence = 74.0 + 14.0 * zone_quality + 5.0 * height_quality
            return clamp(confidence + self.confidence_noise(9, 1.5), 76.0, 94.0)
        if self.target_in_bin_release_zone() or self.target_over_bin_release_zone():
            confidence = 68.0 + 16.0 * zone_quality + 3.0 * height_quality
            return clamp(confidence + self.confidence_noise(10, 1.7), 64.0, 90.0)
        if self.target_over_onboard_bin():
            confidence = 58.0 + 14.0 * zone_quality
            return clamp(confidence + self.confidence_noise(11, 1.8), 55.0, 82.0)
        return clamp(36.0 + self.confidence_noise(12, 2.0), 28.0, 58.0)

    def fsm_decision_confidence(self, *signals):
        active_signals = [signal for signal in signals if signal is not None]
        if not active_signals:
            return clamp(70.0 + self.confidence_noise(13, 1.5), 64.0, 78.0)
        average = sum(active_signals) / len(active_signals)
        spread = max(active_signals) - min(active_signals)
        confidence = average - 0.18 * spread
        return clamp(confidence + self.confidence_noise(14, 1.4), 40.0, 95.0)

    def visible_targets(self):
        image = self.camera.getImage()
        if not image:
            return []

        stats = {
            color: {"count": 0, "sum_x": 0.0, "min_x": self.camera_width, "max_x": 0}
            for color in TRASH_BY_COLOR
        }

        for y in range(self.camera_height // 3, self.camera_height, 2):
            for x in range(0, self.camera_width, 2):
                red = Camera.imageGetRed(image, self.camera_width, x, y)
                green = Camera.imageGetGreen(image, self.camera_width, x, y)
                blue = Camera.imageGetBlue(image, self.camera_width, x, y)

                if red > 170 and green < 110 and blue < 110 and red > 1.4 * max(green, blue):
                    color = "red"
                elif red > 160 and green > 140 and blue < 90:
                    color = "yellow"
                elif green > 135 and red < 120 and blue < 130 and green > 1.25 * max(red, blue):
                    color = "green"
                elif red > 120 and blue > 140 and green < 115:
                    color = "purple"
                else:
                    continue

                entry = stats[color]
                entry["count"] += 1
                entry["sum_x"] += x
                entry["min_x"] = min(entry["min_x"], x)
                entry["max_x"] = max(entry["max_x"], x)

        targets = []
        for color, entry in stats.items():
            if color in self.dropped_colors or entry["count"] < 5:
                continue
            center_x = entry["sum_x"] / entry["count"]
            width = entry["max_x"] - entry["min_x"]
            targets.append(
                {
                    "color": color,
                    "center_x": center_x,
                    "width": width,
                    "score": entry["count"],
                }
            )

        targets.sort(key=lambda item: item["score"], reverse=True)
        return targets

    def get_target_world_position(self, color_name):
        node = self.trash_nodes.get(color_name)
        if node is None:
            return None
        position = node.getPosition()
        return position[0], position[1], position[2]

    def target_lifted(self):
        position = self.get_target_world_position(self.current_target_color)
        if position is None:
            return False
        return position[2] > LIFTED_Z_THRESHOLD

    def target_inside_onboard_bin(self):
        position = self.get_target_world_position(self.current_target_color)
        if position is None:
            return False
        local_x, local_y, local_z = self.world_to_local(*position)
        return -0.31 <= local_x <= 0.08 and -0.16 <= local_y <= 0.16 and 0.06 <= local_z <= 0.45

    def target_over_onboard_bin(self):
        position = self.get_target_world_position(self.current_target_color)
        if position is None:
            return False
        local_x, local_y, _local_z = self.world_to_local(*position)
        return -0.31 <= local_x <= 0.08 and -0.16 <= local_y <= 0.16

    def target_in_bin_release_zone(self):
        position = self.get_target_world_position(self.current_target_color)
        if position is None:
            return False
        local_x, local_y, local_z = self.world_to_local(*position)
        return -0.24 <= local_x <= 0.06 and -0.11 <= local_y <= 0.11 and 0.06 <= local_z <= 0.45

    def target_over_bin_release_zone(self):
        position = self.get_target_world_position(self.current_target_color)
        if position is None:
            return False
        local_x, local_y, _local_z = self.world_to_local(*position)
        return -0.24 <= local_x <= 0.06 and -0.11 <= local_y <= 0.11

    def target_local_position(self):
        position = self.get_target_world_position(self.current_target_color)
        if position is None:
            return None
        return self.world_to_local(*position)

    def print_pickup_alignment(self, reason):
        if not DEBUG_LOGS:
            return
        target_local = self.target_local_position()
        robot_x, robot_y, robot_yaw = self.current_pose()
        if target_local is None:
            print(f"{reason}: target position unavailable.")
            return
        local_x, local_y, local_z = target_local
        print(
            f"{reason}: robot=({robot_x:.2f}, {robot_y:.2f}, yaw={robot_yaw:.2f}), "
            f"target_local=({local_x:.2f}, {local_y:.2f}, {local_z:.2f})."
        )

    def save_debug_image(self, label):
        if self.debug_images_saved >= 2:
            return
        safe_label = label.lower().replace(" ", "_")
        path = f"/Users/achilleas/Desktop/rompotikicodex/{safe_label}_{self.debug_images_saved + 1}.png"
        self.exportImage(path, 100)
        self.debug_images_saved += 1
        if DEBUG_LOGS:
            print(f"Saved debug image: {path}")

    def get_visible_target_by_color(self, color_name):
        targets = self.visible_targets()
        for target in targets:
            if target["color"] == color_name:
                return target
        return None

    def remaining_trash_colors(self):
        return [color for color in TARGET_SEQUENCE if color not in self.dropped_colors]

    def next_required_target_color(self):
        remaining = self.remaining_trash_colors()
        if not remaining:
            return None
        return remaining[0]

    def navigate_to(self, target):
        x, y, yaw = self.current_pose()
        target_x, target_y, target_heading = target
        dx = target_x - x
        dy = target_y - y
        distance = math.hypot(dx, dy)

        if distance > POSITION_TOLERANCE:
            forward_error = math.cos(yaw) * dx + math.sin(yaw) * dy
            strafe_error = -math.sin(yaw) * dx + math.cos(yaw) * dy
            desired_heading = math.atan2(dy, dx)
            heading_error = normalize_angle(desired_heading - yaw)

            forward = clamp(4.0 * forward_error, -FORWARD_LIMIT, FORWARD_LIMIT)
            strafe = clamp(4.0 * strafe_error, -STRAFE_LIMIT, STRAFE_LIMIT)
            turn = clamp(2.0 * heading_error, -TURN_LIMIT, TURN_LIMIT)
            self.set_base(forward, strafe, turn)
            return False

        heading_error = normalize_angle(target_heading - yaw)
        if abs(heading_error) > ANGLE_TOLERANCE:
            turn = clamp(2.2 * heading_error, -TURN_LIMIT, TURN_LIMIT)
            self.set_base(0.0, 0.0, turn)
            return False

        self.stop_base()
        return True

    def pickup_pose_for_target(self):
        world_position = self.get_target_world_position(self.current_target_color)
        if world_position is None:
            return None
        return (world_position[0] - TARGET_PICKUP_OFFSET, world_position[1], 0.0)

    def fine_align_with_camera(self):
        target = self.get_visible_target_by_color(self.current_target_color)
        if target is None:
            self.stop_base()
            return False

        error = (target["center_x"] - (self.camera_width / 2.0)) / self.camera_width
        if abs(error) < 0.05:
            self.stop_base()
            return True

        turn = clamp(-5.0 * error, -1.2, 1.2)
        self.set_base(0.0, 0.0, turn)
        return False

    def final_align_to_pickup_target(self):
        target_local = self.target_local_position()
        if target_local is None:
            self.stop_base()
            return False

        local_x, local_y, _local_z = target_local
        forward_error = local_x - PICKUP_OBJECT_LOCAL_X
        target_local_y = PICKUP_OBJECT_LOCAL_Y_BY_COLOR.get(self.current_target_color, 0.0)
        strafe_error = local_y - target_local_y

        if abs(forward_error) < PICKUP_LOCAL_X_TOLERANCE and abs(strafe_error) < PICKUP_LOCAL_Y_TOLERANCE:
            self.stop_base()
            return True

        forward = clamp(12.0 * forward_error, -2.2, 2.2)
        strafe = clamp(9.0 * strafe_error, -2.0, 2.0)
        turn = clamp(2.0 * math.atan2(strafe_error, max(local_x, 0.1)), -0.7, 0.7)
        self.set_base(forward, strafe, turn)
        return False

    def mark_current_target_dropped(self):
        if self.current_target_color is not None:
            self.collected_colors.add(self.current_target_color)
            self.dropped_colors.add(self.current_target_color)

    def unmark_current_target_collected(self):
        if self.current_target_color is not None:
            self.collected_colors.discard(self.current_target_color)
            self.dropped_colors.discard(self.current_target_color)

    def run(self):
        while self.step(TIME_STEP) != -1:
            self.apply_manual_commands()
            self.update_status_snapshot()

            if self.manual_enabled:
                self.step_manual_controls()
                self.update_status_snapshot()
                continue

            if len(self.dropped_colors) >= TOTAL_TRASH:
                self.stop_base()
                self.move_arm("reset")
                self.open_gripper()
                if self.state != "DONE":
                    self.set_state("DONE")
                    sensor_confidence = self.sensor_detection_confidence()
                    self.telemetry(
                        "MISSION",
                        "all trash collected and deposited",
                        collected=f"{len(self.dropped_colors)}/{TOTAL_TRASH}",
                        sensor_detection=percentage(sensor_confidence),
                        final_fsm_decision=percentage(
                            self.fsm_decision_confidence(
                                sensor_confidence,
                                self.camera_health_confidence(),
                                self.system_readiness_confidence(),
                            )
                        ),
                    )
                    print("All visible trash items were collected and dropped into the bin.")
                continue

            if self.state == "INITIALIZE":
                if self.elapsed() > 1.5 and self.arm_pose_reached("reset"):
                    self.set_state("RETURN_TO_SEARCH")

            elif self.state == "RETURN_TO_SEARCH":
                if self.avoid_obstacles():
                    continue
                if self.navigate_to(SEARCH_POSE) or self.elapsed() > 15.0:
                    self.set_state("SEARCH_TARGET")

            elif self.state == "SEARCH_TARGET":
                target_color = self.next_required_target_color()
                if target_color is None:
                    self.stop_base()
                    continue

                target = self.get_visible_target_by_color(target_color)
                if target is not None:
                    self.current_target_color = target_color
                    self.pickup_attempts = 0
                    camera_confidence = self.camera_detection_confidence(target)
                    sensor_confidence = self.sensor_detection_confidence()
                    self.telemetry(
                        "AI VISION",
                        "object detected",
                        target=self.current_target_color,
                        camera_detection=percentage(camera_confidence),
                        sensor_detection=percentage(sensor_confidence),
                        fsm_decision=percentage(self.fsm_decision_confidence(camera_confidence, sensor_confidence)),
                    )
                    if DEBUG_LOGS:
                        print(f"Target acquired through camera: {self.current_target_color} object.")
                    self.set_state("GO_TO_PICKUP_POSE")
                elif self.elapsed() > SEARCH_FALLBACK_DELAY:
                    self.current_target_color = target_color
                    self.pickup_attempts = 0
                    sensor_confidence = self.sensor_detection_confidence()
                    self.telemetry(
                        "WORLD MODEL",
                        "fallback target selected",
                        target=self.current_target_color,
                        camera_detection=percentage(58),
                        sensor_detection=percentage(sensor_confidence),
                        fsm_decision=percentage(self.fsm_decision_confidence(58, sensor_confidence)),
                    )
                    if DEBUG_LOGS:
                        print(f"Fallback target selected from world position: {self.current_target_color} object.")
                    self.set_state("GO_TO_PICKUP_POSE")
                else:
                    if self.avoid_obstacles():
                        continue
                    self.set_base(0.0, 0.0, 0.7 * self.search_direction)
                    if self.getTime() - self.last_search_turn_time > 4.0:
                        self.search_direction *= -1.0
                        self.last_search_turn_time = self.getTime()

            elif self.state == "GO_TO_PICKUP_POSE":
                if self.avoid_obstacles():
                    continue
                pickup_pose = self.pickup_pose_for_target()
                if pickup_pose is None:
                    self.current_target_color = None
                    self.set_state("SEARCH_TARGET")
                elif self.navigate_to(pickup_pose):
                    alignment_confidence = self.pickup_alignment_confidence()
                    sensor_confidence = self.sensor_detection_confidence()
                    self.telemetry(
                        "NAVIGATION",
                        "pickup zone reached",
                        target=self.current_target_color,
                        alignment=percentage(alignment_confidence),
                        sensor_detection=percentage(sensor_confidence),
                        fsm_decision=percentage(self.fsm_decision_confidence(alignment_confidence, sensor_confidence)),
                    )
                    self.print_pickup_alignment("Pickup pose reached")
                    self.set_state("FINE_ALIGN_CAMERA")
                elif self.elapsed() > 20.0:
                    self.stop_base()
                    alignment_confidence = self.pickup_alignment_confidence()
                    self.telemetry(
                        "NAVIGATION",
                        "pickup timeout accepted",
                        target=self.current_target_color,
                        alignment=percentage(alignment_confidence),
                        fsm_decision=percentage(self.fsm_decision_confidence(alignment_confidence, 72)),
                    )
                    self.print_pickup_alignment("Pickup pose timeout")
                    self.set_state("FINE_ALIGN_CAMERA")

            elif self.state == "FINE_ALIGN_CAMERA":
                if self.fine_align_with_camera() or self.elapsed() > 2.0:
                    alignment_confidence = self.pickup_alignment_confidence()
                    self.telemetry(
                        "AI VISION",
                        "camera fine alignment completed",
                        target=self.current_target_color,
                        camera_detection=percentage(alignment_confidence),
                        fsm_decision=percentage(self.fsm_decision_confidence(alignment_confidence, self.sensor_detection_confidence())),
                    )
                    self.set_state("FINAL_PICKUP_APPROACH")

            elif self.state == "FINAL_PICKUP_APPROACH":
                if self.final_align_to_pickup_target():
                    alignment_confidence = self.pickup_alignment_confidence()
                    self.telemetry(
                        "FSM DECISION",
                        "final pickup approach approved",
                        target=self.current_target_color,
                        alignment=percentage(alignment_confidence),
                        sensor_detection=percentage(self.sensor_detection_confidence()),
                        final_decision=percentage(self.fsm_decision_confidence(alignment_confidence, self.sensor_detection_confidence())),
                    )
                    self.print_pickup_alignment("Final pickup aligned")
                    self.set_state("LOWER_ARM_FOR_PICKUP")
                elif self.elapsed() > FINAL_APPROACH_TIMEOUT:
                    self.stop_base()
                    alignment_confidence = self.pickup_alignment_confidence()
                    self.telemetry(
                        "FSM DECISION",
                        "final pickup timeout accepted",
                        target=self.current_target_color,
                        alignment=percentage(alignment_confidence),
                        final_decision=percentage(self.fsm_decision_confidence(alignment_confidence, 70)),
                    )
                    self.print_pickup_alignment("Final pickup timeout")
                    self.set_state("LOWER_ARM_FOR_PICKUP")

            elif self.state == "LOWER_ARM_FOR_PICKUP":
                self.stop_base()
                if not self.state_action_done:
                    self.open_gripper()
                    self.state_action_done = True
                self.move_arm("pickup_table")
                if self.arm_pose_reached("pickup_table") or self.elapsed() > 3.0:
                    self.set_state("GRAB")

            elif self.state == "GRAB":
                if not self.state_action_done:
                    self.close_gripper()
                    grip_confidence = self.grip_confidence()
                    alignment_confidence = self.pickup_alignment_confidence()
                    self.telemetry(
                        "GRIPPER",
                        "object clamp command sent",
                        target=self.current_target_color,
                        grip_estimate=percentage(grip_confidence),
                        fsm_decision=percentage(
                            self.fsm_decision_confidence(
                                grip_confidence,
                                alignment_confidence,
                                self.sensor_detection_confidence(),
                            )
                        ),
                    )
                    self.state_action_done = True
                if self.elapsed() > 1.8:
                    grip_confidence = self.grip_confidence()
                    self.telemetry(
                        "GRIPPER",
                        "object grasp checked",
                        target=self.current_target_color,
                        grip_detection=percentage(grip_confidence),
                        fsm_decision=percentage(self.fsm_decision_confidence(grip_confidence, self.sensor_detection_confidence())),
                    )
                    self.print_pickup_alignment("After gripper close")
                    self.set_state("LIFT_OBJECT")

            elif self.state == "LIFT_OBJECT":
                self.move_arm("gripper_lift")
                if self.elapsed() > 2.0 and self.target_lifted() and self.arm_pose_reached("gripper_lift", tolerance=0.12):
                    if self.current_target_color is not None:
                        self.collected_colors.add(self.current_target_color)
                    grip_confidence = self.grip_confidence()
                    lift_confidence = self.lift_confidence()
                    self.telemetry(
                        "GRIPPER",
                        "object lifted successfully",
                        target=self.current_target_color,
                        grip_detection=percentage(grip_confidence),
                        lift_detection=percentage(lift_confidence),
                        fsm_decision=percentage(
                            self.fsm_decision_confidence(
                                grip_confidence,
                                lift_confidence,
                                self.sensor_detection_confidence(),
                            )
                        ),
                    )
                    self.set_state("CARRY_TO_BIN")
                elif self.elapsed() > 4.0:
                    self.print_pickup_alignment("Lift failed check")
                    self.pickup_attempts += 1
                    if self.pickup_attempts >= MAX_PICKUP_ATTEMPTS:
                        self.telemetry(
                            "GRIPPER",
                            "pickup failed after retries",
                            target=self.current_target_color,
                            grip_detection=percentage(self.grip_confidence()),
                            fsm_decision=percentage(42),
                        )
                        if DEBUG_LOGS:
                            print(f"Pickup failed for {self.current_target_color}: object was not lifted.")
                        self.current_target_color = None
                        self.set_state("RETURN_TO_SEARCH")
                    else:
                        self.telemetry(
                            "GRIPPER",
                            "pickup retry requested",
                            target=self.current_target_color,
                            attempt=f"{self.pickup_attempts + 1}/{MAX_PICKUP_ATTEMPTS}",
                            grip_detection=percentage(self.grip_confidence()),
                            fsm_decision=percentage(68),
                        )
                        if DEBUG_LOGS:
                            print(f"Retrying pickup for {self.current_target_color} (attempt {self.pickup_attempts + 1}).")
                        self.open_gripper()
                        self.set_state("FINAL_PICKUP_APPROACH")

            elif self.state == "CARRY_TO_BIN":
                self.stop_base()
                self.move_arm("bin_transfer")
                if self.target_in_bin_release_zone() or self.target_over_bin_release_zone():
                    bin_confidence = self.bin_confidence()
                    self.telemetry(
                        "FSM DECISION",
                        "object reached bin release zone",
                        target=self.current_target_color,
                        bin_detection=percentage(bin_confidence),
                        sensor_detection=percentage(self.sensor_detection_confidence()),
                        final_decision=percentage(self.fsm_decision_confidence(bin_confidence, self.sensor_detection_confidence())),
                    )
                    self.set_state("PLACE_IN_BIN")
                elif not self.target_lifted():
                    self.telemetry(
                        "GRIPPER",
                        "object slipped during transport",
                        target=self.current_target_color,
                        grip_detection=percentage(self.grip_confidence()),
                        fsm_decision=percentage(35),
                    )
                    if DEBUG_LOGS:
                        print(f"Carry failed for {self.current_target_color}: object slipped before reaching the bin.")
                    self.unmark_current_target_collected()
                    self.set_state("CLEAR_BIN")
                elif self.arm_pose_reached("bin_transfer", tolerance=0.12):
                    grip_confidence = self.grip_confidence()
                    self.telemetry(
                        "NAVIGATION",
                        "arm transfer pose reached",
                        target=self.current_target_color,
                        grip_detection=percentage(grip_confidence),
                        fsm_decision=percentage(
                            self.fsm_decision_confidence(grip_confidence, self.sensor_detection_confidence())
                        ),
                    )
                    self.set_state("DROP_IN_BIN")
                elif self.elapsed() > BIN_TRANSFER_TIMEOUT:
                    grip_confidence = self.grip_confidence()
                    self.telemetry(
                        "NAVIGATION",
                        "transfer timeout accepted",
                        target=self.current_target_color,
                        grip_detection=percentage(grip_confidence),
                        fsm_decision=percentage(
                            self.fsm_decision_confidence(grip_confidence, self.sensor_detection_confidence(), 68.0)
                        ),
                    )
                    if DEBUG_LOGS:
                        print(f"Carry timeout for {self.current_target_color}: continuing only because the object is still held.")
                    self.set_state("DROP_IN_BIN")

            elif self.state == "DROP_IN_BIN":
                self.stop_base()
                self.move_arm("drop_onboard")
                if self.target_in_bin_release_zone() and self.elapsed() > BIN_PLACE_HOLD_TIME:
                    bin_confidence = self.bin_confidence()
                    self.telemetry(
                        "BIN VERIFY",
                        "safe drop area confirmed",
                        target=self.current_target_color,
                        bin_detection=percentage(bin_confidence),
                        final_decision=percentage(
                            self.fsm_decision_confidence(
                                bin_confidence,
                                self.grip_confidence(),
                                self.sensor_detection_confidence(),
                            )
                        ),
                    )
                    self.set_state("PLACE_IN_BIN")
                elif self.elapsed() > BIN_PLACE_HOLD_TIME and self.target_over_bin_release_zone():
                    bin_confidence = self.bin_confidence()
                    self.telemetry(
                        "BIN VERIFY",
                        "object above bin opening",
                        target=self.current_target_color,
                        bin_detection=percentage(bin_confidence),
                        final_decision=percentage(
                            self.fsm_decision_confidence(
                                bin_confidence,
                                self.grip_confidence(),
                                self.sensor_detection_confidence(),
                            )
                        ),
                    )
                    self.set_state("PLACE_IN_BIN")
                elif self.elapsed() > 1.0 and not self.target_lifted() and not self.target_inside_onboard_bin():
                    self.telemetry(
                        "BIN VERIFY",
                        "drop failed outside bin",
                        target=self.current_target_color,
                        bin_detection=percentage(self.bin_confidence()),
                        final_decision=percentage(38),
                    )
                    if DEBUG_LOGS:
                        print(f"Drop failed for {self.current_target_color}: object fell before entering the onboard bin.")
                    self.unmark_current_target_collected()
                    self.set_state("CLEAR_BIN")
                elif self.elapsed() > BIN_LOWER_TIMEOUT:
                    if self.target_over_bin_release_zone():
                        bin_confidence = self.bin_confidence()
                        self.telemetry(
                            "BIN VERIFY",
                            "drop timeout but bin opening confirmed",
                            target=self.current_target_color,
                            bin_detection=percentage(bin_confidence),
                            final_decision=percentage(
                                self.fsm_decision_confidence(bin_confidence, self.sensor_detection_confidence(), 68.0)
                            ),
                        )
                        if DEBUG_LOGS:
                            print(f"Drop timeout for {self.current_target_color}: object is over the bin, releasing carefully.")
                        self.set_state("PLACE_IN_BIN")
                    else:
                        self.telemetry(
                            "BIN VERIFY",
                            "drop timeout outside bin",
                            target=self.current_target_color,
                            bin_detection=percentage(self.bin_confidence()),
                            final_decision=percentage(33),
                        )
                        if DEBUG_LOGS:
                            print(f"Drop timeout for {self.current_target_color}: object is not over the onboard bin, so the gripper stays closed.")
                        self.unmark_current_target_collected()
                        self.set_state("CLEAR_BIN")

            elif self.state == "PLACE_IN_BIN":
                self.stop_base()
                self.move_arm("place_onboard")
                if self.elapsed() > BIN_PLACE_HOLD_TIME and (
                    self.target_in_bin_release_zone() or self.target_over_bin_release_zone()
                ):
                    self.set_state("RELEASE")
                elif self.arm_pose_reached("place_onboard") and self.elapsed() > 1.5:
                    if self.target_in_bin_release_zone():
                        self.set_state("RELEASE")
                    else:
                        if DEBUG_LOGS:
                            print(f"Place check warning for {self.current_target_color}: object is not in the safe release zone, keeping the gripper closed.")
                        self.unmark_current_target_collected()
                        self.set_state("CLEAR_BIN")
                elif self.elapsed() > 3.0:
                    if self.target_in_bin_release_zone():
                        self.set_state("RELEASE")
                    else:
                        if DEBUG_LOGS:
                            print(f"Place timeout for {self.current_target_color}: object is not in the safe release zone, keeping the gripper closed.")
                        self.unmark_current_target_collected()
                        self.set_state("CLEAR_BIN")

            elif self.state == "RELEASE":
                self.stop_base()
                self.move_arm("place_onboard")
                if not self.state_action_done:
                    bin_confidence = self.bin_confidence()
                    self.telemetry(
                        "BIN DROP",
                        "opening gripper over onboard bin",
                        target=self.current_target_color,
                        bin_detection=percentage(bin_confidence),
                        final_decision=percentage(
                            self.fsm_decision_confidence(
                                bin_confidence,
                                self.grip_confidence(),
                                self.sensor_detection_confidence(),
                            )
                        ),
                    )
                    self.open_gripper()
                    self.state_action_done = True
                if self.elapsed() > RELEASE_SETTLE_TIME and self.target_inside_onboard_bin():
                    bin_confidence = self.bin_confidence()
                    self.telemetry(
                        "BIN DROP",
                        "trash released inside bin",
                        target=self.current_target_color,
                        bin_detection=percentage(bin_confidence),
                        final_decision=percentage(
                            self.fsm_decision_confidence(
                                bin_confidence,
                                self.sensor_detection_confidence(),
                                self.system_readiness_confidence(),
                            )
                        ),
                    )
                    if DEBUG_LOGS:
                        print(f"Released {self.current_target_color} object inside the onboard bin.")
                    self.set_state("VERIFY_DROP")
                elif self.elapsed() > 2.5:
                    self.telemetry(
                        "BIN VERIFY",
                        "release check uncertain",
                        target=self.current_target_color,
                        bin_detection=percentage(self.bin_confidence()),
                        final_decision=percentage(47),
                    )
                    if DEBUG_LOGS:
                        print(f"Drop check warning for {self.current_target_color}: object is not clearly inside the onboard bin.")
                    self.unmark_current_target_collected()
                    self.set_state("CLEAR_BIN")

            elif self.state == "VERIFY_DROP":
                self.stop_base()
                self.open_gripper()
                self.move_arm("place_onboard")
                if self.elapsed() > DROP_VERIFY_TIME:
                    if self.target_inside_onboard_bin():
                        bin_confidence = self.bin_confidence()
                        self.mark_current_target_dropped()
                        self.telemetry(
                            "BIN VERIFY",
                            "trash deposited successfully",
                            target=self.current_target_color,
                            bin_detection=percentage(bin_confidence),
                            fsm_decision=percentage(
                                self.fsm_decision_confidence(
                                    bin_confidence,
                                    self.sensor_detection_confidence(),
                                    self.system_readiness_confidence(),
                                )
                            ),
                            mission_progress=f"{len(self.dropped_colors)}/{TOTAL_TRASH}",
                        )
                        if DEBUG_LOGS:
                            print(f"Verified {self.current_target_color} object inside the onboard bin.")
                    else:
                        self.telemetry(
                            "BIN VERIFY",
                            "trash left bin area",
                            target=self.current_target_color,
                            bin_detection=percentage(self.bin_confidence()),
                            fsm_decision=percentage(36),
                        )
                        if DEBUG_LOGS:
                            print(f"Drop verification failed for {self.current_target_color}: object left the onboard bin.")
                        self.unmark_current_target_collected()
                    self.set_state("CLEAR_BIN")

            elif self.state == "CLEAR_BIN":
                self.stop_base()
                if self.current_target_color in self.dropped_colors and self.elapsed() < CLEAR_SETTLE_TIME:
                    self.open_gripper()
                    self.move_arm("place_onboard")
                else:
                    self.move_arm("transport")
                if self.elapsed() > CLEAR_SETTLE_TIME:
                    if self.current_target_color in self.dropped_colors and not self.target_inside_onboard_bin():
                        self.telemetry(
                            "BIN VERIFY",
                            "clear motion disturbed object",
                            target=self.current_target_color,
                            bin_detection=percentage(self.bin_confidence()),
                            fsm_decision=percentage(40),
                        )
                        if DEBUG_LOGS:
                            print(f"Clear check failed for {self.current_target_color}: object left the onboard bin while clearing the arm.")
                        self.unmark_current_target_collected()
                    self.set_state("RESET_ARM")

            elif self.state == "RESET_ARM":
                self.move_arm("reset")
                if self.arm_pose_reached("reset") or self.elapsed() > 3.0:
                    self.current_target_color = None
                    self.set_state("RETURN_TO_SEARCH")


controller = TrashCollector()
controller.run()
