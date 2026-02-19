extends Node

## Send Full Body Tracking data to VRChat via OSC.
##
## Maps SMPL-24 joint positions received from MeTRAbs to VRChat's
## OSC tracker addresses.  Coordinate conversion from Godot
## (right-handed, Y-up, -Z forward) to Unity/VRChat (left-handed,
## Y-up, +Z forward) is handled automatically.
##
## VRChat OSC Trackers spec:
##   https://docs.vrchat.com/docs/osc-trackers

## VRChat OSC destination IP
@export var vrchat_host: String = "127.0.0.1"
## VRChat OSC destination port (default 9000)
@export var vrchat_port: int = 9000
## Enable / disable OSC sending
@export var enabled: bool = true
## Send rotation estimates derived from bone directions.
## Currently limited to pitch/yaw only (no roll).  Vertical bones
## (hip, chest, head) produce unstable rotations.  Enable with caution.
@export var send_rotation: bool = false

@export_group("Trackers")
## Send head tracker for tracking-space alignment.
## WARNING: enabling this while wearing an HMD will conflict with
## VRChat's normal head tracking.  Only enable for HMD-less mocap.
@export var send_head: bool = false
## Hip tracker (VRChat tracker 1)
@export var send_hip: bool = true
## Left foot tracker (VRChat tracker 3)
@export var send_left_foot: bool = true
## Right foot tracker (VRChat tracker 4)
@export var send_right_foot: bool = true
## Chest tracker (VRChat tracker 2)
@export var send_chest: bool = false
## Left knee tracker (VRChat tracker 5)
@export var send_left_knee: bool = false
## Right knee tracker (VRChat tracker 6)
@export var send_right_knee: bool = false
## Left elbow tracker (VRChat tracker 7)
@export var send_left_elbow: bool = false
## Right elbow tracker (VRChat tracker 8)
@export var send_right_elbow: bool = false

var _udp := PacketPeerUDP.new()
# name -> index into joint_positions array
var _joint_index: Dictionary = {}

# ── Tracker definitions ──────────────────────────────────────────
# id        : VRChat tracker number (1-8, no 0)
# joint     : SMPL-24 joint whose position is sent
# fwd_from  : bone start for rotation estimate (parent)
# fwd_to    : bone end   for rotation estimate (child)
# prop      : @export property name that enables/disables this tracker
const TRACKER_DEFS: Array = [
	# id, joint,  fwd_from, fwd_to,  prop
	{"id": 1, "joint": "pelv", "fwd_from": "pelv", "fwd_to": "spi1", "prop": "send_hip"},
	{"id": 3, "joint": "lank", "fwd_from": "lank", "fwd_to": "ltoe", "prop": "send_left_foot"},
	{"id": 4, "joint": "rank", "fwd_from": "rank", "fwd_to": "rtoe", "prop": "send_right_foot"},
	{"id": 2, "joint": "spi3", "fwd_from": "spi3", "fwd_to": "neck", "prop": "send_chest"},
	{"id": 5, "joint": "lkne", "fwd_from": "lhip", "fwd_to": "lkne", "prop": "send_left_knee"},
	{"id": 6, "joint": "rkne", "fwd_from": "rhip", "fwd_to": "rkne", "prop": "send_right_knee"},
	{"id": 7, "joint": "lelb", "fwd_from": "lsho", "fwd_to": "lelb", "prop": "send_left_elbow"},
	{"id": 8, "joint": "relb", "fwd_from": "rsho", "fwd_to": "relb", "prop": "send_right_elbow"},
]


func _ready() -> void:
	if not enabled:
		return
	var err := _udp.set_dest_address(vrchat_host, vrchat_port)
	if err != OK:
		push_error("VRChat OSC: failed to set dest %s:%d – %s" % [
			vrchat_host, vrchat_port, error_string(err)])
		enabled = false
		return
	print("VRChat OSC: sending FBT to %s:%d" % [vrchat_host, vrchat_port])


## Call every frame with new pose data from MeTRAbs.
## [param positions]: Array of [x,y,z] arrays (Godot space) or null.
## [param names]: Array of joint name strings.
func update_tracking(positions: Array, names: Array) -> void:
	if not enabled:
		return

	# Build name → index mapping once
	if _joint_index.is_empty() and not names.is_empty():
		for i in range(names.size()):
			_joint_index[str(names[i])] = i

	# Head (optional, for tracking-space alignment)
	if send_head:
		_send_head_tracker(positions)

	# Body trackers (send only those enabled via checkboxes)
	for def in TRACKER_DEFS:
		if get(def["prop"]):
			_send_body_tracker(def, positions)


# ── Head tracker ─────────────────────────────────────────────────

func _send_head_tracker(positions: Array) -> void:
	var head_pos = _get_joint(positions, "head")
	if head_pos == null:
		return

	var vrc_pos := _to_vrchat_position(head_pos)
	_send_osc_fff("/tracking/trackers/head/position",
		vrc_pos.x, vrc_pos.y, vrc_pos.z)

	if send_rotation:
		var neck_pos = _get_joint(positions, "neck")
		if neck_pos != null:
			var rot := _estimate_euler(neck_pos, head_pos)
			_send_osc_fff("/tracking/trackers/head/rotation",
				rot.x, rot.y, rot.z)


# ── Body tracker ─────────────────────────────────────────────────

func _send_body_tracker(def: Dictionary, positions: Array) -> void:
	var pos = _get_joint(positions, def["joint"])
	if pos == null:
		return

	var tid := str(def["id"])
	var vrc_pos := _to_vrchat_position(pos)
	_send_osc_fff("/tracking/trackers/" + tid + "/position",
		vrc_pos.x, vrc_pos.y, vrc_pos.z)

	if send_rotation:
		var from = _get_joint(positions, def["fwd_from"])
		var to = _get_joint(positions, def["fwd_to"])
		if from != null and to != null:
			var rot := _estimate_euler(from, to)
			_send_osc_fff("/tracking/trackers/" + tid + "/rotation",
				rot.x, rot.y, rot.z)


# ── Joint helpers ────────────────────────────────────────────────

## Return raw [x,y,z] array for a named joint, or null.
func _get_joint(positions: Array, joint_name: String):
	var idx: int = _joint_index.get(joint_name, -1)
	if idx < 0 or idx >= positions.size():
		return null
	var p = positions[idx]
	if p == null or typeof(p) != TYPE_ARRAY or p.size() < 3:
		return null
	return p


# ── Coordinate conversion ───────────────────────────────────────

## Godot (right-handed, Y-up, -Z fwd) → VRChat/Unity (left-handed, Y-up, +Z fwd).
func _to_vrchat_position(joint: Array) -> Vector3:
	return Vector3(joint[0], joint[1], -joint[2])


## Estimate euler angles (degrees) from a bone direction.
## Returns VRChat-compatible euler (pitch, yaw, roll) for ZXY application order.
func _estimate_euler(from_joint: Array, to_joint: Array) -> Vector3:
	# Direction in Godot space
	var dx: float = to_joint[0] - from_joint[0]
	var dy: float = to_joint[1] - from_joint[1]
	var dz: float = to_joint[2] - from_joint[2]

	var length := sqrt(dx * dx + dy * dy + dz * dz)
	if length < 0.001:
		return Vector3.ZERO

	dx /= length
	dy /= length
	dz /= length

	# Convert direction to VRChat space (negate Z)
	var vrc_dz := -dz

	# Euler angles in VRChat's ZXY application order:
	#   yaw   (Y) = horizontal heading from +Z toward +X
	#   pitch (X) = vertical tilt, positive = looking up
	#   roll  (Z) = 0 (unknown from direction alone)
	var yaw := rad_to_deg(atan2(dx, vrc_dz))
	var pitch := rad_to_deg(asin(clampf(dy, -1.0, 1.0)))
	return Vector3(pitch, yaw, 0.0)


# ── OSC encoding ────────────────────────────────────────────────
# Minimal OSC 1.0 encoder – only handles messages with 3 floats,
# which is all VRChat tracker addresses need.

## Send an OSC message with address and exactly 3 float arguments.
func _send_osc_fff(address: String, f0: float, f1: float, f2: float) -> void:
	var packet := PackedByteArray()

	# 1) Address pattern (null-terminated, padded to 4-byte boundary)
	packet.append_array(address.to_ascii_buffer())
	packet.append(0)
	while packet.size() % 4 != 0:
		packet.append(0)

	# 2) Type tag string: ",fff" (null-terminated, padded)
	packet.append(0x2C)  # ','
	packet.append(0x66)  # 'f'
	packet.append(0x66)  # 'f'
	packet.append(0x66)  # 'f'
	packet.append(0)
	while packet.size() % 4 != 0:
		packet.append(0)

	# 3) Float arguments (big-endian / network order)
	packet.append_array(_float_be(f0))
	packet.append_array(_float_be(f1))
	packet.append_array(_float_be(f2))

	_udp.put_packet(packet)


## Encode a float as 4 bytes in big-endian order.
func _float_be(value: float) -> PackedByteArray:
	var buf := PackedByteArray()
	buf.resize(4)
	buf.encode_float(0, value)
	buf.reverse()
	return buf


func _exit_tree() -> void:
	_udp.close()
