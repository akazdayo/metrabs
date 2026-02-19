extends Node3D

## UDP port to listen on (must match main.py UDP_DEFAULT_PORT)
@export var udp_port: int = 9000
## Radius of joint spheres in meters
@export var joint_radius: float = 0.03
## Color of joint spheres
@export var joint_color: Color = Color(0.2, 0.85, 0.2)
## Color of bone lines
@export var bone_color: Color = Color(0.8, 0.8, 0.8, 0.8)

var udp := PacketPeerUDP.new()
var joint_nodes: Dictionary = {}  # int index -> MeshInstance3D
var joint_edges: Array = []

# Shared resources (created once, reused for all spheres)
var _sphere_mesh: SphereMesh
var _joint_material: StandardMaterial3D

# Line rendering
var _line_mesh_instance: MeshInstance3D
var _immediate_mesh: ImmediateMesh


func _ready() -> void:
	var err := udp.bind(udp_port, "*")
	if err != OK:
		push_error("Failed to bind UDP port %d: %s" % [udp_port, error_string(err)])
		set_process(false)
		return
	print("UDP listening on port %d" % udp_port)

	# Shared sphere mesh
	_sphere_mesh = SphereMesh.new()
	_sphere_mesh.radius = joint_radius
	_sphere_mesh.height = joint_radius * 2.0
	_sphere_mesh.radial_segments = 12
	_sphere_mesh.rings = 6

	# Shared material for joints
	_joint_material = StandardMaterial3D.new()
	_joint_material.albedo_color = joint_color
	_joint_material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED

	# Line renderer for skeleton bones
	_line_mesh_instance = MeshInstance3D.new()
	_immediate_mesh = ImmediateMesh.new()
	_line_mesh_instance.mesh = _immediate_mesh

	var line_material := StandardMaterial3D.new()
	line_material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	line_material.albedo_color = bone_color
	line_material.vertex_color_use_as_albedo = false
	_line_mesh_instance.material_override = line_material

	add_child(_line_mesh_instance)


func _process(_delta: float) -> void:
	# Process only the latest packet (skip stale ones)
	var latest_packet: PackedByteArray = PackedByteArray()
	while udp.get_available_packet_count() > 0:
		latest_packet = udp.get_packet()

	if latest_packet.is_empty():
		return

	var text := latest_packet.get_string_from_utf8()
	var data = JSON.parse_string(text)
	if data == null or typeof(data) != TYPE_DICTIONARY:
		return

	_update_pose(data)


func _update_pose(data: Dictionary) -> void:
	# Update edge topology if provided
	if data.has("joint_edges"):
		joint_edges = data["joint_edges"]

	if not data.has("joint_positions"):
		return

	var positions: Array = data["joint_positions"]

	# Update or create joint spheres
	for i in range(positions.size()):
		if positions[i] == null:
			if joint_nodes.has(i):
				joint_nodes[i].visible = false
			continue

		# Validate that joint entry is an array of 3 numbers
		if typeof(positions[i]) != TYPE_ARRAY or positions[i].size() < 3:
			if joint_nodes.has(i):
				joint_nodes[i].visible = false
			continue

		var pos := Vector3(
			positions[i][0],
			positions[i][1],
			positions[i][2]
		)

		if not joint_nodes.has(i):
			joint_nodes[i] = _create_sphere(i)

		joint_nodes[i].position = pos
		joint_nodes[i].visible = true

	# Draw bone connections
	_draw_bones(positions)


func _create_sphere(index: int) -> MeshInstance3D:
	var sphere := MeshInstance3D.new()
	sphere.name = "joint_%d" % index
	sphere.mesh = _sphere_mesh
	sphere.set_surface_override_material(0, _joint_material)
	add_child(sphere)
	return sphere


func _draw_bones(positions: Array) -> void:
	_immediate_mesh.clear_surfaces()

	if joint_edges.is_empty():
		return

	_immediate_mesh.surface_begin(Mesh.PRIMITIVE_LINES)

	for edge in joint_edges:
		if typeof(edge) != TYPE_ARRAY or edge.size() < 2:
			continue
		var from_idx: int = int(edge[0])
		var to_idx: int = int(edge[1])

		if from_idx >= positions.size() or to_idx >= positions.size():
			continue
		if positions[from_idx] == null or positions[to_idx] == null:
			continue
		if typeof(positions[from_idx]) != TYPE_ARRAY or positions[from_idx].size() < 3:
			continue
		if typeof(positions[to_idx]) != TYPE_ARRAY or positions[to_idx].size() < 3:
			continue

		var from_pos := Vector3(
			positions[from_idx][0],
			positions[from_idx][1],
			positions[from_idx][2]
		)
		var to_pos := Vector3(
			positions[to_idx][0],
			positions[to_idx][1],
			positions[to_idx][2]
		)

		_immediate_mesh.surface_add_vertex(from_pos)
		_immediate_mesh.surface_add_vertex(to_pos)

	_immediate_mesh.surface_end()


func _exit_tree() -> void:
	udp.close()
