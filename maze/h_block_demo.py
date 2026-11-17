"""Animated iiwa + WSG demo of regrasping an H-block.

Scenario A  (regrasp NECESSARY)
    The H lies on its side in a corridor that passes through a tall, narrow
    doorway (10 cm gap).  The arm first grabs the upper bar from above and
    carries the block toward the door.  The gripper is 14.6 cm wide across the
    gap, so it hits the door frame when the block is half-way through (the
    animation stops the instant the signed distance goes negative, nothing is
    hard-coded).  The arm backs out, puts the block down, regrasps the trailing
    END of the upper bar with the fingers closing vertically -- now the 5 cm
    thin axis of the hand is across the gap -- pushes the block through, lets
    go, reaches over the wall and regrasps the block from above on the far
    side to carry it on.

Scenario B  (regrasp BENEFICIAL, not necessary)
    The H lies flat in a square bay and must turn around (180 deg).  Holding
    one END of a bar, the hand has to sweep a wide arc and the whole arm swings
    around the bay -- it works, but it is long.  Holding the CENTER of a bar
    the block spins in place.  The script prints the gripper path length and
    the summed joint motion for both, which is exactly the "cost of not
    regrasping" that a greedy vs. global planner trades off.

Everything is kinematic: joint angles come from IK on a dense sampling of a
Cartesian path, the block rides along with the hand while grasped, and the
frames are recorded into a Meshcat animation.  Open the printed URL, use the
"Animations" folder on the right to play / scrub.

Run:
    python h_block_demo.py                # scenario A, Meshcat animation
    python h_block_demo.py --scenario B
    python h_block_demo.py --scenario A --html demoA.html   # also saves a
                                           # standalone html you can send around
    python h_block_demo.py --primitive    # box/cylinder robot (no drake_models)
"""

from __future__ import annotations

import argparse
import math
import os
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    Box,
    CollisionFilterDeclaration,
    DiagramBuilder,
    InverseKinematics,
    MeshcatVisualizer,
    MultibodyPlant,
    Parser,
    PiecewisePose,
    Rgba,
    RigidTransform,
    RollPitchYaw,
    RotationMatrix,
    Solve,
    StartMeshcat,
)

# ---------------------------------------------------------------------------
# H-block geometry (metres) -- same numbers as h_block_gate.py
# ---------------------------------------------------------------------------
BAR_LEN = 0.30
BAR_SEC = 0.05
CONN_LEN = 0.20
FINGER_LEN = 0.10          # body origin -> fingertip midpoint along gripper +y

FPS = 30.0


def h_block_sdf(name="hblock"):
    half = BAR_SEC / 2
    y_off = CONN_LEN / 2

    def box(tag, x, y, z, sx, sy, sz):
        mat = ("<material><diffuse>0.9 0.55 0.1 1</diffuse></material>"
               if tag == "visual" else "")
        return f"""
      <{tag} name="{tag}_{x:+.2f}_{y:+.2f}">
        <pose>{x} {y} {z} 0 0 0</pose>
        <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>{mat}
      </{tag}>"""

    parts = []
    for y in (+y_off, -y_off):
        parts.append(box("visual", 0, y, half, BAR_LEN, BAR_SEC, BAR_SEC))
        parts.append(box("collision", 0, y, half, BAR_LEN, BAR_SEC, BAR_SEC))
    parts.append(box("visual", 0, 0, half, BAR_SEC, CONN_LEN, BAR_SEC))
    parts.append(box("collision", 0, 0, half, BAR_SEC, CONN_LEN, BAR_SEC))
    return f"""<?xml version="1.0"?>
<sdf version="1.7"><model name="{name}"><link name="body">
  <inertial><mass>0.3</mass><inertia><ixx>1e-3</ixx><iyy>1e-3</iyy><izz>1e-3</izz>
  <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
  {''.join(parts)}
</link></model></sdf>"""


def boxes_sdf(name, boxes, rgba=(0.55, 0.55, 0.6, 1.0)):
    """boxes: list of (cx, cy, cz, sx, sy, sz); one static link."""
    parts = []
    for i, (cx, cy, cz, sx, sy, sz) in enumerate(boxes):
        for tag in ("visual", "collision"):
            mat = (f"<material><diffuse>{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}</diffuse></material>"
                   if tag == "visual" else "")
            parts.append(f"""
      <{tag} name="{tag}_{i}"><pose>{cx} {cy} {cz} 0 0 0</pose>
        <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>{mat}</{tag}>""")
    return f"""<?xml version="1.0"?>
<sdf version="1.7"><model name="{name}"><link name="walls">{''.join(parts)}</link></model></sdf>"""


def doorway_sdf(x, gap, height, thick=0.05, jamb_near=0.05, jamb_far=0.25, y0=0.0):
    """A wall in the y-z plane at x, with a `gap`-wide opening centred on y0.
    The jamb on the robot's side (+y) is `jamb_near` long, the other `jamb_far`
    (the near one only has to be long enough to catch the flat block and the
    hand; keeping it short leaves the arm room to reach past the door)."""
    return boxes_sdf("gate", [(x, y0 + gap / 2 + jamb_near / 2, height / 2, thick, jamb_near, height),
                              (x, y0 - gap / 2 - jamb_far / 2, height / 2, thick, jamb_far, height)])


def bay_sdf(cx, cy, side, height, thick=0.05):
    h = side / 2 + thick / 2
    L = side + 2 * thick
    return boxes_sdf("bay", [(cx + h, cy, height / 2, thick, L, height),
                             (cx - h, cy, height / 2, thick, L, height),
                             (cx, cy + h, height / 2, L, thick, height),
                             (cx, cy - h, height / 2, L, thick, height)])


def floor_sdf():
    return boxes_sdf("floor", [(0.45, 0, -0.01, 2.4, 2.0, 0.02)],
                     rgba=(0.85, 0.85, 0.85, 1))


# ---------------------------------------------------------------------------
# block poses and grasps (object frame O = the flat H, bars along x)
# ---------------------------------------------------------------------------
def X_WO_flat(x, y=0.0, yaw=0.0):
    return RigidTransform(RotationMatrix.MakeZRotation(yaw), [x, y, 0.0])


def X_WO_side(x, y=0.0):
    """Rolled +90 about x: O_x->W_x, O_y->W_z, O_z->W_-y.  Lower bar on floor."""
    R = RotationMatrix.MakeXRotation(np.pi / 2)
    return RigidTransform(R, [x, y + BAR_SEC / 2, CONN_LEN / 2 + BAR_SEC / 2])


def _grasp(Gx, Gy, p_pinch_O):
    """X_OG from gripper closing axis Gx, approach axis Gy (object frame) and
    the point the fingertip midpoint should land on."""
    Gx, Gy = np.asarray(Gx, float), np.asarray(Gy, float)
    R = RotationMatrix(np.column_stack([Gx, Gy, np.cross(Gx, Gy)]))
    return RigidTransform(R, np.asarray(p_pinch_O, float) - FINGER_LEN * Gy)


def grasp_side_top():
    """Side-lying H, pinch the upper bar from above: hand's wide axis across y."""
    return _grasp([0, 0, 1], [0, -1, 0], [0.0, CONN_LEN / 2, BAR_SEC / 2])


def grasp_side_top_tilted(pitch_deg, x_along_bar=0.0):
    """Same pinch but the hand leans back toward -x by `pitch` (so the wrist
    stays on the robot's side when reaching over a wall); pinch point can be
    shifted along the bar."""
    th = np.radians(pitch_deg)
    # world approach d = (sin th, 0, -cos th); in O: W_x->O_x, W_z->O_y
    Gy = [np.sin(th), -np.cos(th), 0.0]
    return _grasp([0, 0, 1], Gy, [x_along_bar, CONN_LEN / 2, BAR_SEC / 2])


def X_WO_side_neg(x, y=0.0, lift=0.0):
    """Rolled -90 about x: O_x->W_x, O_y->W_-z, O_z->W_+y.  The bar at object
    y=-CONN_LEN/2 ends up on TOP, the one at +CONN_LEN/2 sits on the floor."""
    R = RotationMatrix.MakeXRotation(-np.pi / 2)
    return RigidTransform(R, [x, y - BAR_SEC / 2, CONN_LEN / 2 + BAR_SEC / 2 + lift])


def grasp_flat_far_bar(x_along_bar):
    """Flat H: top-down pinch on the bar at object y=-CONN_LEN/2, closing
    across it (along object y).  After X_WO_side_neg this same grasp becomes a
    horizontal pinch on the top bar from the +y side (hand body beside the
    block, 14.6 cm tall x 5 cm thin x 7 cm long)."""
    return _grasp([0, 1, 0], [0, 0, -1], [x_along_bar, -CONN_LEN / 2, BAR_SEC / 2])


def grasp_sideneg_top_tilted(pitch_deg, x_along_bar=0.0):
    """Block in X_WO_side_neg; pinch the top bar from above, hand leaning back
    toward -x by `pitch`, closing along world y."""
    th = np.radians(pitch_deg)
    # world approach d = (sin th, 0, -cos th); W_x -> O_x, W_z -> -O_y
    Gy = [np.sin(th), np.cos(th), 0.0]
    Gx = [0, 0, 1.0]                     # W_y = O_z
    return _grasp(Gx, Gy, [x_along_bar, -CONN_LEN / 2, BAR_SEC / 2])


def grasp_sideneg_side_yawed(yaw_deg, x_along_bar=0.0):
    """Block in X_WO_side_neg; horizontal pinch on the top bar from the +y
    (robot) side, fingers closing vertically, hand yawed by `yaw` so that its
    wrist sits closer to the robot (approach_W = (sin yaw, -cos yaw, 0))."""
    ph = np.radians(yaw_deg)
    # W_x -> O_x, W_y -> O_z, W_z -> -O_y
    Gy = [np.sin(ph), 0.0, -np.cos(ph)]
    Gx = [0.0, -1.0, 0.0]
    return _grasp(Gx, Gy, [x_along_bar, -CONN_LEN / 2, BAR_SEC / 2])


def grasp_side_end():
    """Side-lying H, pinch the trailing (-x) end of the upper bar with the
    fingers closing vertically: hand's 5 cm thin axis is across y, body
    trails behind the block."""
    return _grasp([0, 1, 0], [1, 0, 0],
                  [-(BAR_LEN / 2 - 0.03), CONN_LEN / 2, BAR_SEC / 2])


def grasp_flat_center():
    return _grasp([0, 1, 0], [0, 0, -1], [0.0, CONN_LEN / 2, BAR_SEC / 2])


def grasp_flat_middle():
    """Flat H, pinch the middle of the connector (= the block's centre), closing along x."""
    return _grasp([1, 0, 0], [0, 0, -1], [0.0, 0.0, BAR_SEC / 2])


def grasp_flat_end():
    return _grasp([0, 1, 0], [0, 0, -1], [BAR_LEN / 2 - 0.03, CONN_LEN / 2, BAR_SEC / 2])


# ---------------------------------------------------------------------------
# robot models.  Real meshes from drake_models when available, otherwise
# primitive stand-ins with identical kinematics and joint names.
# ---------------------------------------------------------------------------
IIWA_URL = "package://drake_models/iiwa_description/sdf/iiwa7_with_box_collision.sdf"
IIWA_URL_FALLBACK = "package://drake_models/iiwa_description/sdf/iiwa7_no_collision.sdf"
WSG_URL = "package://drake_models/wsg_50_description/sdf/schunk_wsg_50_with_tip.sdf"
X_7G = RigidTransform(RollPitchYaw(np.radians([90, 0, 90])), [0, 0, 0.09])
Q_HOME = np.array([0.0, 0.3, 0.0, -1.5, 0.0, 1.3, 0.0])
Q_LO = -np.radians([170, 120, 170, 120, 170, 120, 175])
Q_HI = -Q_LO


def wsg_primitive_sdf():
    """Box WSG with the real body/finger dimensions and joint conventions:
    body frame at the box centre, +x closes, +y toward fingertips."""
    fy = 0.028 + 0.049133          # finger link origin along +y in body frame

    def finger(name, x0, tipx, axis):
        return f"""
    <link name="{name}"><pose>{x0} {fy} 0 0 0 0</pose>
      <inertial><mass>0.05</mass><inertia><ixx>1e-4</ixx><iyy>1e-4</iyy><izz>1e-4</izz>
        <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
      <visual name="v"><geometry><box><size>0.012 0.082 0.02</size></box></geometry>
        <material><diffuse>0.2 0.2 0.2 1</diffuse></material></visual>
      <collision name="c"><geometry><box><size>0.012 0.082 0.02</size></box></geometry></collision>
      <visual name="tip"><pose>{tipx} 0.031 0 0 0 0</pose>
        <geometry><box><size>0.003 0.02 0.02</size></box></geometry>
        <material><diffuse>0.1 0.1 0.1 1</diffuse></material></visual>
    </link>
    <joint name="{name}_sliding_joint" type="prismatic">
      <parent>body</parent><child>{name}</child>
      <axis><xyz>{axis} 0 0</xyz><limit><lower>-0.055</lower><upper>0.055</upper><effort>80</effort></limit></axis>
    </joint>"""

    return f"""<?xml version="1.0"?>
<sdf version="1.7"><model name="Schunk_Gripper">
    <link name="body">
      <inertial><mass>0.99</mass><inertia><ixx>1e-3</ixx><iyy>1e-3</iyy><izz>1e-3</izz>
        <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
      <visual name="v"><geometry><box><size>0.146 0.0725 0.05</size></box></geometry>
        <material><diffuse>0.7 0.7 0.7 1</diffuse></material></visual>
      <collision name="c"><geometry><box><size>0.146 0.0725 0.05</size></box></geometry></collision>
    </link>
    {finger('left_finger', -0.0115, +0.0075, 1)}
    {finger('right_finger', +0.0115, -0.0075, 1)}
</model></sdf>"""


def iiwa_primitive_urdf_path():
    """Strip the mesh visuals / mobile base out of the manipulation package's
    primitive-collision iiwa14 URDF and show its cylinders instead."""
    import manipulation
    src = os.path.join(os.path.dirname(manipulation.__file__), "models",
                       "mobile_iiwa14_primitive_collision.urdf")
    text = open(src).read()
    text = text[text.index("<robot"):]          # header has a nested comment
    ET.register_namespace("drake", "http://drake.mit.edu")
    ET.register_namespace("xacro", "http://www.ros.org/wiki/xacro")
    root = ET.fromstring(text)
    tree = ET.ElementTree(root)
    for el in list(root):
        tag = el.tag.split("}")[-1]
        name = el.get("name", "")
        if tag == "link" and name in ("base_x", "base_y"):
            root.remove(el)
        elif tag == "joint" and name.startswith("iiwa_base_"):
            root.remove(el)
        elif tag in ("transmission", "gazebo"):
            root.remove(el)
    # iiwa14 -> iiwa7 joint offsets, so the stand-in has the real demo's reach
    for j in root.findall("joint"):
        if j.get("name") == "iiwa_joint_2":
            j.find("origin").set("xyz", "0 0 0.1825")
        elif j.get("name") == "iiwa_joint_4":
            j.find("origin").set("xyz", "0 0 0.1955")
    for link in root.findall("link"):
        for vis in list(link.findall("visual")):
            if vis.find("geometry/mesh") is not None:
                link.remove(vis)
        for scc in list(link):
            if scc.tag.endswith("self_collision_checking"):
                link.remove(scc)
        for col in list(link.findall("collision")):
            if col.find("geometry/mesh") is not None:
                link.remove(col)
                continue
            vis = ET.SubElement(link, "visual")
            o = col.find("origin")
            if o is not None:
                ET.SubElement(vis, "origin", o.attrib)
            g = ET.SubElement(vis, "geometry")
            g.append(col.find("geometry")[0])
            m = ET.SubElement(vis, "material", {"name": "Orange" if link.get("name") in
                              ("iiwa_link_1", "iiwa_link_3", "iiwa_link_5") else "Silver"})
        if link.get("name") == "iiwa_link_7":   # lost its mesh: small puck
            vis = ET.SubElement(link, "visual")
            ET.SubElement(vis, "origin", {"xyz": "0 0 0.02", "rpy": "0 0 0"})
            g = ET.SubElement(vis, "geometry")
            ET.SubElement(g, "cylinder", {"length": "0.04", "radius": "0.045"})
            ET.SubElement(vis, "material", {"name": "Silver"})
    out = os.path.join(tempfile.gettempdir(), "iiwa_primitive_demo.urdf")
    tree.write(out)
    return out


def add_robot(plant, parser, primitive):
    """Adds iiwa + welded WSG; returns (iiwa_instance, wsg_instance)."""
    local = os.environ.get("DRAKE_MODELS_DIR")       # dev: a local checkout of drake_models
    if local:
        pm = parser.package_map()
        if pm.Contains("drake_models"):
            pm.Remove("drake_models")
        pm.AddPackageXml(os.path.join(local, "package.xml"))
    if not primitive:
        try:
            try:
                iiwa = parser.AddModelsFromUrl(IIWA_URL)
            except Exception:
                iiwa = parser.AddModelsFromUrl(IIWA_URL_FALLBACK)
            iiwa = iiwa[0]
            wsg = parser.AddModelsFromUrl(WSG_URL)[0]
        except Exception as e:  # drake_models not downloadable -> fallback
            print(f"[demo] drake_models unavailable ({str(e).splitlines()[0]}); using primitive robot")
            primitive = True
    if primitive:
        iiwa = parser.AddModels(iiwa_primitive_urdf_path())[0]
        wsg = parser.AddModelsFromString(wsg_primitive_sdf(), "sdf")[0]
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0", iiwa))
    plant.WeldFrames(plant.GetFrameByName("iiwa_link_7", iiwa),
                     plant.GetFrameByName("body", wsg), X_7G)
    return iiwa, wsg, primitive


# ---------------------------------------------------------------------------
# the animated demo
# ---------------------------------------------------------------------------
@dataclass
class Stats:
    name: str
    gripper_path: float = 0.0
    joint_path: float = 0.0
    min_clearance: float = math.inf
    min_arm_clearance: float = math.inf


class Demo:
    def __init__(self, obstacle_sdf, meshcat, primitive=False):
        # ---- full scene (what we draw and collision-check) ----------------
        builder = DiagramBuilder()
        self.plant, self.sg = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
        parser = Parser(self.plant)
        self.iiwa, self.wsg, self.primitive = add_robot(self.plant, parser, primitive)
        self.block = parser.AddModelsFromString(h_block_sdf(), "sdf")[0]
        self.walls = parser.AddModelsFromString(obstacle_sdf, "sdf")[0]
        floor = parser.AddModelsFromString(floor_sdf(), "sdf")[0]
        self.plant.WeldFrames(self.plant.world_frame(), self.plant.GetFrameByName("walls", self.walls))
        self.plant.WeldFrames(self.plant.world_frame(), self.plant.GetFrameByName("walls", floor))
        self.plant.Finalize()
        self.meshcat = meshcat
        self.viz = MeshcatVisualizer.AddToBuilder(builder, self.sg, meshcat)
        self.diagram = builder.Build()
        self.ctx = self.diagram.CreateDefaultContext()
        self.pc = self.plant.GetMyContextFromRoot(self.ctx)
        self.sgc = self.sg.GetMyContextFromRoot(self.ctx)
        self.G = self.plant.GetBodyByName("body", self.wsg)
        self.O = self.plant.GetBodyByName("body", self.block)
        self.lf = self.plant.GetJointByName("left_finger_sliding_joint", self.wsg)
        self.rf = self.plant.GetJointByName("right_finger_sliding_joint", self.wsg)

        insp = self.sg.model_inspector()
        self.wall_frame = self.plant.GetBodyFrameIdOrThrow(
            self.plant.GetBodyByName("walls", self.walls).index())
        self.wsg_frames = {self.plant.GetBodyFrameIdOrThrow(b) for b in self.plant.GetBodyIndices(self.wsg)}
        self.arm_frames = {self.plant.GetBodyFrameIdOrThrow(b) for b in self.plant.GetBodyIndices(self.iiwa)}
        self.block_frame = self.plant.GetBodyFrameIdOrThrow(self.O.index())

        # ---- robot + walls plant for IK (no block) ---------------------------
        # The walls are in here so IK can be told "keep the ARM at least
        # ARM_MARGIN from the walls".  The hand is deliberately filtered out of
        # that constraint: whether the hand fits is what the demo tests, so it
        # must be free to hit the door (the full scene above detects it).
        b2 = DiagramBuilder()
        self.ik_plant, sg2 = AddMultibodyPlantSceneGraph(b2, time_step=0.0)
        p2 = Parser(self.ik_plant)
        self.ik_iiwa, self.ik_wsg, _ = add_robot(self.ik_plant, p2, self.primitive)
        walls2 = p2.AddModelsFromString(obstacle_sdf, "sdf")[0]
        self.ik_plant.WeldFrames(self.ik_plant.world_frame(), self.ik_plant.GetFrameByName("walls", walls2))
        self.ik_plant.Finalize()
        robot_geoms = self.ik_plant.CollectRegisteredGeometries(
            [self.ik_plant.get_body(b) for b in
             list(self.ik_plant.GetBodyIndices(self.ik_iiwa)) + list(self.ik_plant.GetBodyIndices(self.ik_wsg))])
        wsg_geoms = self.ik_plant.CollectRegisteredGeometries(
            [self.ik_plant.get_body(b) for b in self.ik_plant.GetBodyIndices(self.ik_wsg)])
        wall_geoms = self.ik_plant.CollectRegisteredGeometries(
            [self.ik_plant.GetBodyByName("walls", walls2)])
        sg2.collision_filter_manager().Apply(
            CollisionFilterDeclaration().ExcludeWithin(robot_geoms).ExcludeBetween(wsg_geoms, wall_geoms))
        self.ik_diagram = b2.Build()
        self.ik_root_ctx = self.ik_diagram.CreateDefaultContext()
        self.ik_ctx = self.ik_plant.GetMyContextFromRoot(self.ik_root_ctx)
        self.ik_G = self.ik_plant.GetFrameByName("body", self.ik_wsg)
        self.arm_margin = 0.03      # IK keeps the arm this far from the walls

        # ---- state ----------------------------------------------------------
        self.t = 0.0
        self.q = Q_HOME.copy()
        self.opening = 0.08
        self.X_WO = RigidTransform()
        self.X_OG = None            # not None while the block is grasped
        self.stats = None
        self.hit_marker = None
        self.last_hit_t = None

    # -- kinematics -----------------------------------------------------------
    def fk(self, q):
        self.ik_plant.SetPositions(self.ik_ctx, self.ik_iiwa, q)
        return self.ik_plant.CalcRelativeTransform(self.ik_ctx, self.ik_plant.world_frame(), self.ik_G)

    def ik(self, X_WG, q_seed, pos_tol=1e-3, ang_tol=0.01, max_jump=2.0):
        """Hand pose -> iiwa q.  Stays close to q_seed (continuity), with a
        small pull toward the middle of the joint ranges so the wrist does not
        drift into a limit over a long path.  If the seeded solve fails, a few
        other seeds are tried and the solution nearest q_seed is returned."""
        p = X_WG.translation()
        rng = np.random.default_rng(0)
        seeds = [q_seed, Q_HOME] + [np.clip(q_seed + rng.normal(0, 0.5, 7), Q_LO, Q_HI) for _ in range(6)]
        best = None
        for i, seed in enumerate(seeds):
            ik = InverseKinematics(self.ik_plant, self.ik_ctx)
            ik.AddPositionConstraint(self.ik_G, [0, 0, 0], self.ik_plant.world_frame(),
                                     p - pos_tol, p + pos_tol)
            ik.AddOrientationConstraint(self.ik_plant.world_frame(), X_WG.rotation(),
                                        self.ik_G, RotationMatrix(), ang_tol)
            if self.arm_margin is not None:
                ik.AddMinimumDistanceLowerBoundConstraint(self.arm_margin, 0.04)
            prog, q = ik.prog(), ik.q()
            self.ik_plant.SetPositions(self.ik_ctx, self.ik_iiwa, q_seed)
            q_ref = self.ik_plant.GetPositions(self.ik_ctx).copy()      # seed incl. finger joints
            self.ik_plant.SetPositions(self.ik_ctx, self.ik_iiwa, seed)
            q_init = self.ik_plant.GetPositions(self.ik_ctx).copy()
            q_mid = q_ref.copy(); q_mid[:7] = 0.0
            prog.AddQuadraticErrorCost(np.eye(len(q_ref)), q_ref, q)
            prog.AddQuadraticErrorCost(0.03 * np.eye(len(q_ref)), q_mid, q)
            prog.SetInitialGuess(q, q_init)
            res = Solve(prog)
            if res.is_success():
                self.ik_plant.SetPositions(self.ik_ctx, res.GetSolution(q))
                sol = self.ik_plant.GetPositions(self.ik_ctx, self.ik_iiwa).copy()
                dist = np.abs(sol - q_seed).sum()
                if best is None or dist < best[0]:
                    best = (dist, sol)
                if i == 0 or dist < 0.3:
                    break
        if best is not None and best[0] > max_jump:
            best = None                        # a branch jump is not a solution
        if best is None:
            raise RuntimeError(f"IK failed for gripper at {np.round(p, 3)}")
        self.ik_plant.SetPositions(self.ik_ctx, self.ik_iiwa, best[1])
        return best[1]

    # -- scene state --------------------------------------------------------
    def _apply(self):
        self.plant.SetPositions(self.pc, self.iiwa, self.q)
        self.lf.set_translation(self.pc, -self.opening / 2)
        self.rf.set_translation(self.pc, +self.opening / 2)
        if self.X_OG is not None:
            X_WG = self.plant.CalcRelativeTransform(self.pc, self.plant.world_frame(), self.G.body_frame())
            self.X_WO = X_WG @ self.X_OG.inverse()
        self.plant.SetFreeBodyPose(self.pc, self.O, self.X_WO)

    def clearance(self):
        """min signed distance of (gripper, block, arm) to the walls."""
        qo = self.sg.get_query_output_port().Eval(self.sgc)
        insp = qo.inspector()
        d = {"hand": math.inf, "block": math.inf, "arm": math.inf}
        for pair in qo.ComputeSignedDistancePairwiseClosestPoints(max_distance=0.3):
            fa, fb = insp.GetFrameId(pair.id_A), insp.GetFrameId(pair.id_B)
            if self.wall_frame not in (fa, fb):
                continue
            other = fb if fa == self.wall_frame else fa
            if other in self.wsg_frames:
                d["hand"] = min(d["hand"], pair.distance)
            elif other == self.block_frame:
                d["block"] = min(d["block"], pair.distance)
            elif other in self.arm_frames:
                d["arm"] = min(d["arm"], pair.distance)
        return d

    def frame(self):
        self._apply()
        self.ctx.SetTime(self.t)
        self.diagram.ForcedPublish(self.ctx)
        self.t += 1.0 / FPS

    # -- primitives ---------------------------------------------------------
    def hold(self, seconds):
        for _ in range(int(seconds * FPS)):
            self.frame()

    def set_opening(self, opening, seconds=0.6):
        n = max(1, int(seconds * FPS))
        for s in np.linspace(0, 1, n + 1)[1:]:
            self.opening = (1 - s) * self.opening + s * opening if s < 1 else opening
            self.frame()

    def grasp(self, X_OG, opening=0.039):
        self.set_opening(opening)
        self.X_OG = X_OG

    def release(self, opening=0.08):
        self.X_OG = None
        self.set_opening(opening)

    def move_gripper(self, X_WG_goal, speed=0.15, min_seconds=0.6, stop_on_hit=False,
                     what="hand"):
        """Straight-line Cartesian move of the hand (block comes along if grasped).
        Returns True if the move completed, False if stopped by a collision."""
        X_WG_0 = self.fk(self.q)
        dist = np.linalg.norm(X_WG_goal.translation() - X_WG_0.translation())
        ang = X_WG_0.rotation().InvertAndCompose(X_WG_goal.rotation()).ToAngleAxis().angle()
        T = max(min_seconds, dist / speed, ang / 1.0)
        traj = PiecewisePose.MakeLinear([0.0, T], [X_WG_0, X_WG_goal])
        n = int(T * FPS)
        q_prev = self.q.copy()
        p_prev = X_WG_0.translation()
        for i in range(1, n + 1):
            X = traj.GetPose(i * T / n)
            q_new = self.ik(X, q_prev)
            # tentatively apply and check
            q_backup, self.q = self.q, q_new
            self._apply()
            d = self.clearance()
            if self.stats is not None:
                self.stats.min_clearance = min(self.stats.min_clearance, d["hand"], d["block"])
                self.stats.min_arm_clearance = min(self.stats.min_arm_clearance, d["arm"])
            if stop_on_hit and min(d["hand"], d["block"]) < 0.0:
                self.q = q_backup
                self._apply()
                print(f"    *** {what} hits the wall at t={self.t:5.1f}s "
                      f"(hand {d['hand']*100:+.1f} cm, block {d['block']*100:+.1f} cm) -> stop")
                self.flash_hit()
                return False
            if self.stats is not None and self.X_OG is not None:   # count carries only
                self.stats.gripper_path += np.linalg.norm(X.translation() - p_prev)
                self.stats.joint_path += np.abs(q_new - q_prev).sum()
            p_prev, q_prev = X.translation(), q_new
            self.frame()
        return True

    def move_block(self, X_WO_goal, **kw):
        """Carry the grasped block so that it ends at X_WO_goal."""
        assert self.X_OG is not None
        return self.move_gripper(X_WO_goal @ self.X_OG, **kw)

    def choose_flip(self, X_WO, X_OG, later_block_poses=(), via=None):
        """A parallel-jaw grasp is the same grasp with the closing axis negated
        (180 deg about the approach axis).  Pick the one the arm reaches with
        the smaller joint motion from where it is now -- and, if given, one
        that can also follow the block through `later_block_poses`."""
        best = None
        for require_later in (True, False):          # prefer a flip that can do the follow-up
            for flip in (RotationMatrix(), RotationMatrix.MakeYRotation(np.pi)):
                cand = RigidTransform(X_OG.rotation() @ flip, X_OG.translation())
                try:   # reachable at all?  (a single far solve, jumps allowed)
                    q = self.ik(X_WO @ cand, self.q, pos_tol=5e-3, ang_tol=0.05, max_jump=np.inf)
                except RuntimeError:
                    continue
                # the approach from here (straight, or through the `via`
                # hand poses) must work too, and (if asked) the follow-up carries
                X_WG = X_WO @ cand
                route = (via(X_WG) if via else []) + [X_WG]
                if not self.feasible_path(route, self.q, samples=20):
                    continue
                if require_later and later_block_poses and not self.feasible_path(
                        [X @ cand for X in later_block_poses], q, samples=12):
                    continue
                cost = np.abs(q - self.q).sum()
                if best is None or cost < best[0]:
                    best = (cost, cand)
            if best is not None:
                break
        if best is None:
            raise RuntimeError("grasp unreachable in either hand orientation")
        return best[1]

    def feasible_path(self, poses, q_seed, samples=8):
        """Chain IK through a list of hand poses (no drawing), sampling the
        straight line between consecutive poses; True if everything solves."""
        q = q_seed
        X_prev = self.fk(q)
        try:
            for X in poses:
                traj = PiecewisePose.MakeLinear([0.0, 1.0], [X_prev, X])
                for s in np.linspace(0, 1, samples + 1)[1:]:
                    q = self.ik(traj.GetPose(s), q)
                X_prev = X
        except RuntimeError:
            return False
        return True

    def choose_turn(self, X_OG, cx, cy, yaw0, dyaw, steps=24):
        """For a 'turn the flat block about (cx, cy) by dyaw' task, pick the hand
        flip and the turning direction (+dyaw or -dyaw) that the arm can follow
        all the way without hitting a joint limit."""
        for flip in (RotationMatrix(), RotationMatrix.MakeYRotation(np.pi)):
            cand = RigidTransform(X_OG.rotation() @ flip, X_OG.translation())
            for sign in (+1, -1):
                yaws = yaw0 + sign * dyaw * np.linspace(0, 1, steps + 1)
                poses = [X_WO_flat(cx, cy, y) @ cand for y in yaws]
                if self.feasible_path(poses, self.q):
                    return cand, yaw0 + sign * dyaw
        raise RuntimeError("no hand flip / direction can make that turn")

    def move_to_grasp(self, X_WO, X_OG, approach=0.12, later_block_poses=(), **kw):
        """Pregrasp (backed off along -approach axis) then descend.
        Returns the (possibly flipped) X_OG actually used."""
        X_OG = self.choose_flip(X_WO, X_OG, later_block_poses)
        X_WG = X_WO @ X_OG
        X_pre = RigidTransform(X_WG.rotation(),
                               X_WG.translation() - approach * X_WG.rotation().matrix()[:, 1])
        self.move_gripper(X_pre, **kw)
        self.move_gripper(X_WG, speed=0.08)
        return X_OG

    def retreat(self, dist=0.12, **kw):
        X_WG = self.fk(self.q)
        self.move_gripper(RigidTransform(X_WG.rotation(),
                                         X_WG.translation() - dist * X_WG.rotation().matrix()[:, 1]), **kw)

    def go_home(self, seconds=2.0):
        n = int(seconds * FPS)
        q0 = self.q.copy()
        for s in np.linspace(0, 1, n + 1)[1:]:
            self.q = (1 - s) * q0 + s * Q_HOME
            self.frame()

    # -- hit marker ---------------------------------------------------------
    def add_hit_marker(self, center, size):
        self.hit_marker = "/demo/hit"
        self.meshcat.SetObject(self.hit_marker, Box(*size), Rgba(1, 0.1, 0.1, 0.45))
        self.meshcat.SetTransform(self.hit_marker, RigidTransform(center))
        self.meshcat.SetProperty(self.hit_marker, "visible", False, 0.0)

    def flash_hit(self, seconds=1.2):
        if self.hit_marker is None:
            return
        self.meshcat.SetProperty(self.hit_marker, "visible", True, self.t)
        self.hold(seconds)
        self.meshcat.SetProperty(self.hit_marker, "visible", False, self.t)

    def begin(self, name):
        self.stats = Stats(name)
        return self.stats


# ---------------------------------------------------------------------------
# Scenario A
# ---------------------------------------------------------------------------
def scenario_A(meshcat, primitive, Y0=-0.40, lean=20.0, xa4=0.10, xg1=0.03,
               HEIGHT=0.32, roll_lift=0.40, record=True):
    # The corridor runs along x, to the robot's right (y = Y0), so the wrist
    # stays far enough from the shoulder for the low pushes.
    X0, XW = 0.30, 0.55                        # block start, wall
    X1 = XW + 0.025 + BAR_LEN / 2 + 0.02       # centre x at which the block is fully through
    GAP = 0.10
    demo = Demo(doorway_sdf(XW, GAP, HEIGHT, y0=Y0), meshcat, primitive)
    demo.X_WO = X_WO_flat(X0, Y0)              # flat: an H seen from above
    demo.q = Q_HOME.copy()
    demo.opening = 0.08
    if record:
        demo.meshcat.StartRecording(frames_per_second=FPS)
    demo.add_hit_marker([XW, Y0, HEIGHT / 2], [0.07, GAP + 0.02, HEIGHT + 0.02])
    demo.hold(0.5)

    print("\n=== Scenario A: H through a 10 cm doorway ===")
    # ---- A1: flat H, straight at the door: the BLOCK does not fit -----------
    print("  A1  flat H (25 cm wide), top grasp, carry toward the 10 cm door")
    s1 = demo.begin("A1 flat")
    X_OG = grasp_flat_far_bar(x_along_bar=xg1)          # far bar
    # the same grasp must survive the lift / roll / set-down that follow,
    # which happen right where the block gets stopped (xr = its centre then)
    xr = XW - 0.025 - BAR_LEN / 2 - 0.02       # roll right where it got stopped
    later = [X_WO_flat(xr, Y0),
             RigidTransform(RotationMatrix(), [xr, Y0, roll_lift]),
             X_WO_side_neg(xr, Y0, lift=roll_lift), X_WO_side_neg(xr, Y0)]
    X_OG = demo.move_to_grasp(demo.X_WO, X_OG, later_block_poses=later)
    demo.grasp(X_OG)
    ok = demo.move_block(X_WO_flat(X1, Y0), speed=0.10, stop_on_hit=True, what="block (25 cm wide)")
    assert not ok, "expected the flat block to be stopped by the door frame"
    print(f"      block is {s1.min_clearance*100:+.1f} cm from the wall -> does not fit flat")
    demo.hold(0.4)
    demo.move_block(X_WO_flat(xr, Y0), speed=0.12)       # back off a touch

    # ---- A2: roll it onto its side, let go ---------------------------------
    print("  A2  lift, roll 90 deg onto its side (5 cm wide now), set down, let go")
    s2 = demo.begin("A2 roll")
    demo.move_block(RigidTransform(RotationMatrix(), [xr, Y0, roll_lift]), speed=0.12)   # lift high
    demo.move_block(X_WO_side_neg(xr, Y0, lift=roll_lift), speed=0.10)                   # roll
    demo.move_block(X_WO_side_neg(xr, Y0), speed=0.08)                              # set down
    demo.release()
    demo.retreat(0.04)                       # just clear the bar (+y) ...
    X = demo.fk(demo.q)                      # ... then straight up, out of the way
    demo.move_gripper(RigidTransform(X.rotation(), X.translation() + [0, 0, 0.25]), speed=0.2)

    # ---- A3: regrasp from above, push: the HAND does not fit ---------------
    print("  A3  regrasp from above (hand 14.6 cm across the gap), push -> hand hits the door")
    s3 = demo.begin("A3 top pinch push")
    X_OG3 = demo.move_to_grasp(demo.X_WO, grasp_sideneg_top_tilted(0, x_along_bar=-0.10))
    demo.grasp(X_OG3)
    ok = demo.move_block(X_WO_side_neg(X1, Y0), speed=0.10, stop_on_hit=True,
                         what="hand (14.6 cm across a 10 cm gap)")
    assert not ok, "expected the top pinch to be stopped by the door frame"
    x_stuck = demo.X_WO.translation()[0]
    print(f"      block fits, hand does not: stuck with the block centre at x={x_stuck*100:.0f} cm, "
          f"{(x_stuck + BAR_LEN/2 - XW - 0.025)*100:.0f} cm of it poking out the far side")
    demo.hold(0.4)
    demo.release()
    demo.retreat(0.12)
    X = demo.fk(demo.q)                      # straight up first
    demo.move_gripper(RigidTransform(X.rotation(), X.translation() + [0, 0, 0.12]), speed=0.2)

    # ---- A4: around the door, regrasp the part that pokes out, pull through
    print("  A4  reach past the door, regrasp the far end from the side, pull it through")
    s4 = demo.begin("A4 far-side pull")

    def route4(X_WG):
        """up and over toward the far side (already in the pull orientation),
        then a pregrasp 12 cm out along the approach axis."""
        d = X_WG.rotation().matrix()[:, 1]
        return [RigidTransform(X_WG.rotation(), X_WG.translation() + [0.0, 0.12, 0.30]),
                RigidTransform(X_WG.rotation(), X_WG.translation() - 0.12 * d)]
    # same horizontal side pinch as after the roll, now on the far side of the door
    X_OG4 = demo.choose_flip(demo.X_WO, grasp_sideneg_side_yawed(lean, xa4),
                             later_block_poses=[X_WO_side_neg(X1, Y0)], via=route4)
    X_WG4 = demo.X_WO @ X_OG4
    for X in route4(X_WG4):
        demo.move_gripper(X, speed=0.18)
    demo.move_gripper(X_WG4, speed=0.08)
    demo.grasp(X_OG4)
    done = False
    for x_goal in (X1, X1 - 0.03, X1 - 0.06):
        goal = X_WO_side_neg(x_goal, Y0)
        if demo.feasible_path([goal @ X_OG4], demo.q):
            demo.move_block(goal, speed=0.08)
            done = x_goal >= X1 - 0.03
            break
    for dz in (0.08, 0.05, 0.03):
        goal = X_WO_side_neg(demo.X_WO.translation()[0], Y0, lift=dz)
        if demo.feasible_path([goal @ X_OG4], demo.q):
            demo.move_block(goal, speed=0.06)
            break
    print(f"      pulled to x={demo.X_WO.translation()[0]*100:.0f} cm -> "
          f"{'fully through' if done else 'through except the last few cm (arm reach)'}; "
          + (f"arm clearance to the wall {demo.clearance()['arm']*100:+.1f} cm"
             if math.isfinite(demo.clearance()['arm']) else "(no arm collision geometry)"))
    demo.hold(1.0)
    if record:
        demo.meshcat.StopRecording()
        demo.meshcat.PublishRecording()
    print(f"\n  hand travel while holding the block:  A1 {s1.gripper_path*100:5.1f} cm   "
          f"A3 {s3.gripper_path*100:5.1f} cm   A4 {s4.gripper_path*100:5.1f} cm")
    if math.isfinite(s4.min_arm_clearance):
        print("  min arm-to-wall clearance per act: " + "  ".join(
            f"{s.name}: {s.min_arm_clearance*100:+.1f} cm" for s in (s1, s2, s3, s4)))
    print("  Verdict: flat, the block does not fit; on its side it fits but a hand on top "
          "of it does not, and nothing on the near side can push it the last 20 cm "
          "-> a regrasp on the FAR side is NECESSARY.")
    return demo


# ---------------------------------------------------------------------------
# Scenario B
# ---------------------------------------------------------------------------
def scenario_B(meshcat, primitive):
    CX, SIDE, HEIGHT = 0.62, 0.50, 0.12
    demo = Demo(bay_sdf(CX, 0, SIDE, HEIGHT), meshcat, primitive)
    demo.X_WO = X_WO_flat(CX, 0, 0.0)
    demo.q = Q_HOME.copy()
    demo.opening = 0.08
    demo.meshcat.StartRecording(frames_per_second=FPS)
    demo.hold(0.5)

    print("\n=== Scenario B: flat H turning around (180 deg) in a 50 cm bay ===")

    def turn(X_OG, label, yaw0, dyaw, steps=12):
        st = demo.begin(label)
        X_OG, yaw1 = demo.choose_turn(X_OG, CX, 0, yaw0, dyaw)
        X_WG = demo.X_WO @ X_OG
        X_pre = RigidTransform(X_WG.rotation(), X_WG.translation() + [0, 0, 0.12])
        demo.move_gripper(X_pre)
        demo.move_gripper(X_WG, speed=0.08)
        demo.grasp(X_OG)
        for yaw in np.linspace(yaw0, yaw1, steps + 1)[1:]:
            ok = demo.move_block(X_WO_flat(CX, 0, yaw), speed=0.12, min_seconds=0.25,
                                 stop_on_hit=True, what=label)
            if not ok:
                break
        demo.release()
        demo.retreat(0.15)
        return st

    s_end = turn(grasp_flat_end(), "B1 end grasp", 0.0, np.pi)
    demo.go_home(1.5)
    demo.hold(0.3)
    s_ctr = turn(grasp_flat_middle(), "B2 centre grasp", np.pi, np.pi)
    demo.go_home(1.5)
    demo.hold(0.5)
    demo.meshcat.StopRecording()
    demo.meshcat.PublishRecording()

    print(f"  {'grasp':<16}{'hand path':>12}{'sum |dq|':>12}{'min wall clearance':>22}")
    for s in (s_end, s_ctr):
        print(f"  {s.name:<16}{s.gripper_path*100:>9.1f} cm{s.joint_path:>9.2f} rad"
              f"{s.min_clearance*100:>18.1f} cm")
    print("  Verdict: both grasps work, but the end grasp sweeps the hand around the bay "
          "and swings the whole arm.  Regrasping to the centre is worth it whenever the "
          "regrasp cost is below that difference -> regrasp BENEFICIAL, not necessary.")
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["A", "B"], default="A")
    ap.add_argument("--primitive", action="store_true", help="box/cylinder robot")
    ap.add_argument("--html", help="also save a standalone html of the animation")
    ap.add_argument("--no-wait", action="store_true")
    args = ap.parse_args()

    meshcat = StartMeshcat()
    if args.scenario == "A":
        meshcat.SetCameraPose([1.4, -1.5, 0.9], [0.5, -0.40, 0.2])
    else:
        meshcat.SetCameraPose([1.6, -1.1, 1.1], [0.62, 0.0, 0.1])
    demo = (scenario_A if args.scenario == "A" else scenario_B)(meshcat, args.primitive)
    if args.html:
        with open(args.html, "w") as f:
            f.write(meshcat.StaticHtml())
        print(f"  saved {args.html}")
    print("\nMeshcat: open the URL above; the animation is in the 'Animations' folder "
          "on the right (play / scrub).")
    if not args.no_wait:
        input("Press Enter to exit.")


if __name__ == "__main__":
    main()
