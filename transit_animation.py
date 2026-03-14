"""
Transit Flow Animation — Astoria Wing Station
Vasilij Pavlov, MArch 2025, Parsons

Generates a 30-second MP4 (or GIF fallback) showing how the proposed
Astoria Wing Station transforms NYC transit from Manhattan-centric
routing to direct interborough connections.
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.patches import Polygon, FancyArrowPatch
import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────

FPS = 30
DURATION = 30
TOTAL_FRAMES = FPS * DURATION  # 900

PHASES = {
    'establish':  (0,   90),
    'act1':       (90,  390),
    'transition': (390, 480),
    'act2':       (480, 780),
    'outro':      (780, 900),
}

# Colors
BG_COLOR      = '#0D1117'
BOROUGH_FILL  = '#141A22'
BOROUGH_EDGE  = '#2A3A4A'
NODE_DEFAULT  = '#FFFFFF'
NODE_HUB_BEFORE = '#FFD700'
NODE_HUB_AFTER  = '#4ECDC4'
FLOW_BEFORE   = '#FF6B6B'
FLOW_AFTER    = '#4ECDC4'
FLOW_AIRPORT  = '#45B7D1'
TEXT_PRIMARY   = '#FFFFFF'
TEXT_SECONDARY = '#8B9BAB'

# RGBA tuples for particle colors
CORAL_RGBA  = (1.0, 0.42, 0.42, 1.0)
TEAL_RGBA   = (0.31, 0.80, 0.77, 1.0)
BLUE_RGBA   = (0.27, 0.72, 0.82, 1.0)

# ── Geographic Data ──────────────────────────────────────────────────────────

# Note: matplotlib y-axis is bottom-up, so we invert the prompt's top-left
# origin coordinates by doing y = 1080 - y_prompt

def _flip(coords):
    """Convert top-left origin coords to bottom-left (matplotlib)."""
    if isinstance(coords, dict):
        return {k: (x, 1080 - y) for k, (x, y) in coords.items()}
    return [(x, 1080 - y) for x, y in coords]

NODES_RAW = {
    "Penn Station":       (680, 520),
    "Grand Central":      (730, 490),
    "Manhattan Core":     (700, 510),
    "Astoria-Ditmars":    (890, 430),
    "Jamaica":            (1050, 560),
    "LGA Airport":        (920, 400),
    "Atlantic Terminal":  (820, 650),
    "Harlem-125th":       (760, 390),
    "Newark":             (540, 510),
    "New Haven":          (1050, 280),
    "Philadelphia":       (580, 780),
    "Long Island":        (1180, 520),
    "Westchester":        (820, 310),
}
NODES = _flip(NODES_RAW)

BOROUGHS_RAW = {
    "Manhattan": [(660,380),(710,360),(760,390),(750,560),(700,600),(650,560),(640,450)],
    "Queens":    [(810,380),(1080,360),(1150,520),(1050,640),(870,640),(800,520),(820,430)],
    "Brooklyn":  [(700,600),(820,580),(870,700),(800,760),(680,720),(660,640)],
    "Bronx":     [(710,360),(820,300),(900,350),(880,430),(800,430),(750,390)],
    "NJ":        [(540,420),(640,400),(650,560),(600,640),(500,600),(480,500)],
}
BOROUGHS = {k: _flip(v) for k, v in BOROUGHS_RAW.items()}

# ── Edge Definitions ─────────────────────────────────────────────────────────

# Current state: everything funnels INTO Manhattan hubs
# Format: (origin, destination) — straight lines converging on gold Manhattan nodes
CURRENT_EDGES_IN = [
    # Inbound to Manhattan
    ("Philadelphia",     "Penn Station",    "NEC inbound"),
    ("Newark",           "Penn Station",    "NJ Transit inbound"),
    ("Long Island",      "Penn Station",    "LIRR inbound"),
    ("New Haven",        "Grand Central",   "Metro-North inbound"),
    ("LGA Airport",      "Manhattan Core",  "LGA via subway"),
    ("Atlantic Terminal", "Manhattan Core",  "Brooklyn inbound"),
    ("Astoria-Ditmars",  "Manhattan Core",  "N/W inbound"),
    ("Harlem-125th",     "Grand Central",   "Bronx inbound"),
    ("Westchester",      "Grand Central",   "Westchester inbound"),
    ("Jamaica",          "Penn Station",    "Jamaica inbound"),
]

# Manhattan hubs also radiate outbound (showing it as THE hub for everything)
CURRENT_EDGES_OUT = [
    ("Penn Station",    "Philadelphia",     "NEC outbound"),
    ("Penn Station",    "Newark",           "NJ Transit outbound"),
    ("Penn Station",    "Long Island",      "LIRR outbound"),
    ("Grand Central",   "New Haven",        "Metro-North outbound"),
    ("Grand Central",   "Westchester",      "Westchester outbound"),
    ("Manhattan Core",  "Atlantic Terminal", "Brooklyn outbound"),
]

# Proposed state: direct connections to/from Astoria
PROPOSED_EDGES = [
    # Inbound to Astoria
    ("Philadelphia",     "Astoria-Ditmars", "teal",    "Amtrak NEC"),
    ("New Haven",        "Astoria-Ditmars", "teal",    "NEC direct"),
    ("LGA Airport",      "Astoria-Ditmars", "airport", "SkyTrain"),
    ("Long Island",      "Astoria-Ditmars", "teal",    "LIRR direct"),
    ("Harlem-125th",     "Astoria-Ditmars", "teal",    "Metro-North"),
    ("Newark",           "Astoria-Ditmars", "teal",    "NJ Transit"),
    # Outbound from Astoria to other boroughs
    ("Astoria-Ditmars",  "Atlantic Terminal", "teal",   "IBX Brooklyn"),
    ("Astoria-Ditmars",  "Harlem-125th",     "teal",   "Bronx direct"),
    ("Astoria-Ditmars",  "Jamaica",           "teal",   "Queens local"),
]

# ── Utility Functions ────────────────────────────────────────────────────────

def quadratic_bezier(t, p0, p1, p2):
    """Vectorized quadratic bezier. t: (N,), p0/p1/p2: (2,). Returns (N,2)."""
    t = np.asarray(t)
    t2 = t[:, np.newaxis]
    p0, p1, p2 = np.asarray(p0), np.asarray(p1), np.asarray(p2)
    return (1 - t2)**2 * p0 + 2 * (1 - t2) * t2 * p1 + t2**2 * p2


def ease_in_out(t):
    """Smooth S-curve easing."""
    t = np.clip(t, 0, 1)
    return np.where(t < 0.5, 4*t*t*t, 1 - (-2*t + 2)**3 / 2)


def ease_out(t):
    return 1 - (1 - np.clip(t, 0, 1))**3


def particle_alpha(t):
    """Fade particles in/out at path endpoints."""
    fade_in  = np.clip(t / 0.1, 0, 1)
    fade_out = np.clip((1 - t) / 0.1, 0, 1)
    return fade_in * fade_out


# ── TransitAnimation ─────────────────────────────────────────────────────────

class TransitAnimation:
    def __init__(self):
        self.rng = np.random.default_rng(42)

        # Figure
        self.fig, self.ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
        self.ax.set_xlim(0, 1920)
        self.ax.set_ylim(0, 1080)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self.fig.patch.set_facecolor(BG_COLOR)
        self.ax.set_facecolor(BG_COLOR)
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Build layers
        self._build_boroughs()
        self._build_edge_lines()
        self._build_nodes()
        self._build_particles()
        self._build_text()

    # ── Borough polygons ──────────────────────────────────────────────────

    def _build_boroughs(self):
        self.borough_patches = []
        for name, verts in BOROUGHS.items():
            p = Polygon(verts, closed=True,
                        facecolor=BOROUGH_FILL, edgecolor=BOROUGH_EDGE,
                        linewidth=1.2, alpha=0, zorder=1)
            self.ax.add_patch(p)
            self.borough_patches.append(p)

    # ── Edge path lines ───────────────────────────────────────────────────

    def _make_arc_path(self, origin_key, dest_key):
        """Create a bezier path with a gentle arc between two nodes."""
        p0 = np.array(NODES[origin_key])
        p2 = np.array(NODES[dest_key])
        mid = (p0 + p2) / 2
        perp = np.array([-(p2[1] - p0[1]), p2[0] - p0[0]])
        norm = np.linalg.norm(perp)
        if norm > 0:
            perp = perp / norm
        p1 = mid + perp * 35  # gentle arc
        return p0, p1, p2

    def _build_edge_lines(self):
        """Pre-compute bezier paths for all edges and draw faint guide lines."""
        # Act 1 paths: inbound + outbound, all converging on/from Manhattan
        self.act1_paths = []
        self.act1_lines = []
        for origin, dest, label in CURRENT_EDGES_IN + CURRENT_EDGES_OUT:
            p0, p1, p2 = self._make_arc_path(origin, dest)
            self.act1_paths.append((p0, p1, p2))
            t_line = np.linspace(0, 1, 60)
            pts = quadratic_bezier(t_line, p0, p1, p2)
            line, = self.ax.plot(pts[:, 0], pts[:, 1],
                                 color=FLOW_BEFORE, linewidth=1.2,
                                 alpha=0, zorder=2)
            self.act1_lines.append(line)

        # Act 2 paths (direct to/from Astoria)
        self.act2_paths = []
        self.act2_lines = []
        self.act2_colors = []
        for origin, dest, ckey, label in PROPOSED_EDGES:
            p0, p1, p2 = self._make_arc_path(origin, dest)
            self.act2_paths.append((p0, p1, p2))
            self.act2_colors.append(BLUE_RGBA if ckey == "airport" else TEAL_RGBA)
            t_line = np.linspace(0, 1, 60)
            pts = quadratic_bezier(t_line, p0, p1, p2)
            color = FLOW_AIRPORT if ckey == "airport" else FLOW_AFTER
            line, = self.ax.plot(pts[:, 0], pts[:, 1],
                                 color=color, linewidth=1.2,
                                 alpha=0, zorder=2)
            self.act2_lines.append(line)

    # ── Station nodes ─────────────────────────────────────────────────────

    def _build_nodes(self):
        # Separate hub nodes from regular ones
        self.hub_names = {"Penn Station", "Grand Central", "Manhattan Core"}
        self.astoria_name = "Astoria-Ditmars"

        names = list(NODES.keys())
        xs = [NODES[n][0] for n in names]
        ys = [NODES[n][1] for n in names]

        # Base node scatter
        self.node_scatter = self.ax.scatter(
            xs, ys, s=40, c=NODE_DEFAULT, zorder=5, alpha=0, edgecolors='none')

        # Hub highlights (Manhattan)
        hub_xs = [NODES[n][0] for n in self.hub_names]
        hub_ys = [NODES[n][1] for n in self.hub_names]
        self.hub_scatter = self.ax.scatter(
            hub_xs, hub_ys, s=80, c=NODE_HUB_BEFORE,
            zorder=6, alpha=0, edgecolors='none')

        # Astoria highlight
        ax, ay = NODES[self.astoria_name]
        self.astoria_scatter = self.ax.scatter(
            [ax], [ay], s=60, c=NODE_HUB_AFTER,
            zorder=6, alpha=0, edgecolors='none')
        # Glow ring
        self.astoria_glow = self.ax.scatter(
            [ax], [ay], s=400, c='none',
            edgecolors=NODE_HUB_AFTER, linewidths=2,
            zorder=4, alpha=0)

        # Labels
        self.node_labels = []
        label_offset = {
            "Philadelphia": (0, -22), "New Haven": (0, 18),
            "Newark": (-10, -22), "Long Island": (0, -22),
            "Westchester": (0, 18), "LGA Airport": (0, 18),
            "Harlem-125th": (-10, 18),
            "Penn Station": (-20, -22), "Grand Central": (20, 18),
            "Manhattan Core": (-40, -22),
            "Astoria-Ditmars": (10, 18),
            "Jamaica": (0, -22), "Atlantic Terminal": (0, -22),
        }
        for name in names:
            x, y = NODES[name]
            dx, dy = label_offset.get(name, (0, -18))
            display = name.replace("Manhattan Core", "").strip()
            if not display:
                self.node_labels.append(None)
                continue
            t = self.ax.text(x + dx, y + dy, display,
                             color=TEXT_SECONDARY, fontsize=7,
                             ha='center', va='center',
                             fontfamily='sans-serif', alpha=0, zorder=7)
            self.node_labels.append(t)

    # ── Particles ─────────────────────────────────────────────────────────

    def _build_particles(self):
        n_per_edge = 10

        # Act 1 particles
        n1 = len(self.act1_paths) * n_per_edge
        self.p1_t = self.rng.uniform(0, 1, n1)
        self.p1_edge = np.repeat(np.arange(len(self.act1_paths)), n_per_edge)
        self.p1_speed = self.rng.uniform(0.006, 0.012, n1)
        self.p1_scatter = self.ax.scatter(
            np.zeros(n1), np.zeros(n1), s=12,
            c=[CORAL_RGBA] * n1, zorder=8, alpha=0, edgecolors='none')

        # Act 2 particles
        n2 = len(self.act2_paths) * n_per_edge
        self.p2_t = self.rng.uniform(0, 1, n2)
        self.p2_edge = np.repeat(np.arange(len(self.act2_paths)), n_per_edge)
        self.p2_speed = self.rng.uniform(0.009, 0.016, n2)  # faster
        colors2 = []
        for i in range(len(self.act2_paths)):
            colors2.extend([self.act2_colors[i]] * n_per_edge)
        self.p2_base_colors = np.array(colors2)
        self.p2_scatter = self.ax.scatter(
            np.zeros(n2), np.zeros(n2), s=12,
            c=colors2, zorder=8, alpha=0, edgecolors='none')

    # ── Text overlays ─────────────────────────────────────────────────────

    def _build_text(self):
        # Title
        self.title_text = self.ax.text(
            120, 1020, "NEW YORK CITY TRANSIT FLOWS",
            color=TEXT_PRIMARY, fontsize=22, fontweight='bold',
            fontfamily='sans-serif', alpha=0, zorder=10)
        self.subtitle_text = self.ax.text(
            120, 985, "Current State vs. All Aboard to Astoria",
            color=TEXT_SECONDARY, fontsize=13,
            fontfamily='sans-serif', alpha=0, zorder=10)

        # Phase subtitles
        self.act1_label = self.ax.text(
            120, 945, "TODAY: All intercity passengers funnel through Manhattan",
            color=FLOW_BEFORE, fontsize=12,
            fontfamily='sans-serif', alpha=0, zorder=10)

        self.transition_label = self.ax.text(
            120, 945, "PROPOSED: Astoria Wing Station as intermodal hub",
            color=FLOW_AFTER, fontsize=12,
            fontfamily='sans-serif', alpha=0, zorder=10)

        # Congestion stat
        mx, my = NODES["Manhattan Core"]
        self.congestion_label = self.ax.text(
            mx - 100, my - 45, "~750,000 daily transfers",
            color=NODE_HUB_BEFORE, fontsize=9,
            fontfamily='sans-serif', alpha=0, zorder=10,
            fontstyle='italic')

        # Astoria stat
        asx, asy = NODES["Astoria-Ditmars"]
        self.astoria_label = self.ax.text(
            asx + 30, asy + 40, "~2M annual Amtrak riders\n+ LGA connectivity",
            color=FLOW_AFTER, fontsize=9,
            fontfamily='sans-serif', alpha=0, zorder=10,
            fontstyle='italic')

        # Edge callout labels for Act 2
        self.callout_labels = []
        callout_data = [
            ("Philadelphia", "Astoria-Ditmars", "Amtrak NEC"),
            ("LGA Airport", "Astoria-Ditmars", "SkyTrain"),
            ("Astoria-Ditmars", "Atlantic Terminal", "IBX Brooklyn"),
            ("Astoria-Ditmars", "Harlem-125th", "Bronx direct"),
        ]
        for origin, dest, text in callout_data:
            mx = (NODES[origin][0] + NODES[dest][0]) / 2
            my = (NODES[origin][1] + NODES[dest][1]) / 2
            t = self.ax.text(mx + 15, my + 10, text,
                             color=TEXT_SECONDARY, fontsize=8,
                             fontfamily='sans-serif', alpha=0, zorder=10,
                             fontstyle='italic',
                             bbox=dict(boxstyle='round,pad=0.2',
                                       facecolor=BG_COLOR, edgecolor='none',
                                       alpha=0.7))
            self.callout_labels.append(t)

        # Outro stats panel
        self.outro_texts = []
        panel_lines = [
            (960, 400, "BEFORE", TEXT_SECONDARY, 14, 'bold'),
            (960, 370, "All intercity travel", FLOW_BEFORE, 11, 'normal'),
            (960, 345, "routes through Manhattan", FLOW_BEFORE, 11, 'normal'),
            (960, 300, "AFTER", TEXT_SECONDARY, 14, 'bold'),
            (960, 270, "Direct outer-borough", FLOW_AFTER, 11, 'normal'),
            (960, 245, "connections via Astoria", FLOW_AFTER, 11, 'normal'),
            (960, 195, "LGA: 45+ min via subway  -->  ~12 min via SkyTrain",
             TEXT_PRIMARY, 11, 'normal'),
            (960, 165, "NEC to Queens: 2 transfers  -->  direct",
             TEXT_PRIMARY, 11, 'normal'),
        ]
        for x, y, text, color, size, weight in panel_lines:
            t = self.ax.text(x, y, text, color=color, fontsize=size,
                             fontweight=weight, ha='center',
                             fontfamily='sans-serif', alpha=0, zorder=10)
            self.outro_texts.append(t)

        # Credit
        self.credit_text = self.ax.text(
            1800, 30,
            "All Aboard to Astoria\nVasilij Pavlov, MArch 2025, Parsons",
            color=TEXT_SECONDARY, fontsize=9, ha='right', va='bottom',
            fontfamily='sans-serif', alpha=0, zorder=10)

    # ── Phase logic ───────────────────────────────────────────────────────

    def _get_phase(self, frame):
        for name, (start, end) in PHASES.items():
            if start <= frame < end:
                progress = (frame - start) / (end - start)
                return name, progress
        return 'outro', 1.0

    def _advance_particles(self, t_arr, speed_arr):
        """Advance particle t values and wrap around."""
        t_arr += speed_arr
        t_arr[t_arr >= 1.0] -= 1.0

    def _compute_positions(self, t_arr, edge_indices, paths):
        """Compute particle positions from t values and paths."""
        positions = np.zeros((len(t_arr), 2))
        for i, (p0, p1, p2) in enumerate(paths):
            mask = edge_indices == i
            if np.any(mask):
                pts = quadratic_bezier(t_arr[mask], p0, p1, p2)
                positions[mask] = pts
        return positions

    # ── Update ────────────────────────────────────────────────────────────

    def _update(self, frame):
        phase, progress = self._get_phase(frame)

        if frame % 100 == 0:
            print(f"  Frame {frame}/{TOTAL_FRAMES} [{phase}]")

        # ── Establish ──
        if phase == 'establish':
            # Fade in boroughs
            ba = float(ease_out(np.clip(progress * 2, 0, 1)))
            for p in self.borough_patches:
                p.set_alpha(ba * 0.8)

            # Fade in nodes
            na = float(ease_out(np.clip((progress - 0.3) / 0.4, 0, 1)))
            self.node_scatter.set_alpha(na * 0.7)

            # Fade in hub nodes
            ha = float(ease_out(np.clip((progress - 0.4) / 0.3, 0, 1)))
            self.hub_scatter.set_alpha(ha)

            # Fade in labels
            la = float(ease_out(np.clip((progress - 0.5) / 0.4, 0, 1)))
            for lbl in self.node_labels:
                if lbl:
                    lbl.set_alpha(la * 0.7)

            # Fade in title
            ta = float(ease_out(np.clip((progress - 0.6) / 0.4, 0, 1)))
            self.title_text.set_alpha(ta)
            self.subtitle_text.set_alpha(ta * 0.8)

        # ── Act 1 ──
        elif phase == 'act1':
            # Ensure base elements visible
            for p in self.borough_patches:
                p.set_alpha(0.8)
            self.node_scatter.set_alpha(0.7)
            self.hub_scatter.set_alpha(1.0)
            self.title_text.set_alpha(1.0)
            self.subtitle_text.set_alpha(0.8)

            # Show act1 label
            self.act1_label.set_alpha(float(ease_out(np.clip(progress * 3, 0, 1))))

            # Edge lines fade in
            line_alpha = float(ease_out(np.clip(progress * 2, 0, 1))) * 0.35
            for line in self.act1_lines:
                line.set_alpha(line_alpha)

            # Advance and draw particles
            self._advance_particles(self.p1_t, self.p1_speed)
            pos = self._compute_positions(self.p1_t, self.p1_edge, self.act1_paths)
            self.p1_scatter.set_offsets(pos)

            # Per-particle alpha
            p_alpha = particle_alpha(self.p1_t)
            ramp = float(np.clip(progress * 3, 0, 1))
            colors = np.tile(np.array(CORAL_RGBA), (len(self.p1_t), 1))
            colors[:, 3] = p_alpha * ramp
            self.p1_scatter.set_facecolors(colors)
            self.p1_scatter.set_alpha(1.0)

            # Manhattan hub pulse
            pulse = 80 + 40 * np.sin(frame * 0.08)
            self.hub_scatter.set_sizes([pulse] * len(self.hub_names))

            # Congestion label
            self.congestion_label.set_alpha(
                float(ease_out(np.clip((progress - 0.2) / 0.3, 0, 1))))

            # Labels visible
            for lbl in self.node_labels:
                if lbl:
                    lbl.set_alpha(0.7)

        # ── Transition ──
        elif phase == 'transition':
            # Fade out act1 elements
            fade_out_a = float(1.0 - ease_out(np.clip(progress * 2, 0, 1)))

            # Act1 particles fade
            self._advance_particles(self.p1_t, self.p1_speed)
            pos1 = self._compute_positions(self.p1_t, self.p1_edge, self.act1_paths)
            self.p1_scatter.set_offsets(pos1)
            p_alpha1 = particle_alpha(self.p1_t) * fade_out_a
            colors1 = np.tile(np.array(CORAL_RGBA), (len(self.p1_t), 1))
            colors1[:, 3] = p_alpha1
            self.p1_scatter.set_facecolors(colors1)

            # Fade edge lines
            for line in self.act1_lines:
                line.set_alpha(0.35 * fade_out_a)

            # Fade labels
            self.act1_label.set_alpha(fade_out_a)
            self.congestion_label.set_alpha(fade_out_a)

            # Hub scatter dims
            self.hub_scatter.set_alpha(fade_out_a)
            self.hub_scatter.set_sizes([80] * len(self.hub_names))

            # Transition label fades in
            fade_in_a = float(ease_out(np.clip((progress - 0.3) / 0.5, 0, 1)))
            self.transition_label.set_alpha(fade_in_a)

            # Astoria begins to glow
            astoria_a = float(ease_out(np.clip((progress - 0.2) / 0.6, 0, 1)))
            self.astoria_scatter.set_alpha(astoria_a)
            glow_size = 400 + 200 * np.sin(frame * 0.1)
            self.astoria_glow.set_sizes([glow_size])
            self.astoria_glow.set_alpha(astoria_a * 0.4)

            # Start act2 particles late in transition
            if progress > 0.5:
                act2_ramp = float((progress - 0.5) / 0.5)
                self._advance_particles(self.p2_t, self.p2_speed)
                pos2 = self._compute_positions(self.p2_t, self.p2_edge, self.act2_paths)
                self.p2_scatter.set_offsets(pos2)
                p_alpha2 = particle_alpha(self.p2_t) * act2_ramp * 0.5
                colors2 = self.p2_base_colors.copy()
                colors2[:, 3] = p_alpha2
                self.p2_scatter.set_facecolors(colors2)
                self.p2_scatter.set_alpha(1.0)

        # ── Act 2 ──
        elif phase == 'act2':
            # Keep base map visible
            for p in self.borough_patches:
                p.set_alpha(0.8)
            self.node_scatter.set_alpha(0.7)
            self.title_text.set_alpha(1.0)
            self.subtitle_text.set_alpha(0.8)
            self.transition_label.set_alpha(1.0)
            self.act1_label.set_alpha(0)
            self.congestion_label.set_alpha(0)

            # Dim Manhattan hubs
            self.hub_scatter.set_alpha(0.25)
            self.hub_scatter.set_sizes([60] * len(self.hub_names))

            # Ensure act1 hidden
            colors1_off = np.tile(np.array(CORAL_RGBA), (len(self.p1_t), 1))
            colors1_off[:, 3] = 0
            self.p1_scatter.set_facecolors(colors1_off)
            for line in self.act1_lines:
                line.set_alpha(0)

            # Astoria glow
            self.astoria_scatter.set_alpha(1.0)
            glow_size = 500 + 150 * np.sin(frame * 0.1)
            self.astoria_glow.set_sizes([glow_size])
            self.astoria_glow.set_alpha(0.5)
            # Larger astoria node
            self.astoria_scatter.set_sizes([180])

            # Edge lines fade in
            line_alpha = float(ease_out(np.clip(progress * 3, 0, 1))) * 0.35
            for line in self.act2_lines:
                line.set_alpha(line_alpha)

            # Advance and draw act2 particles
            self._advance_particles(self.p2_t, self.p2_speed)
            pos2 = self._compute_positions(self.p2_t, self.p2_edge, self.act2_paths)
            self.p2_scatter.set_offsets(pos2)
            ramp = float(np.clip(progress * 3, 0, 1))
            p_alpha2 = particle_alpha(self.p2_t) * ramp
            colors2 = self.p2_base_colors.copy()
            colors2[:, 3] = p_alpha2
            self.p2_scatter.set_facecolors(colors2)
            self.p2_scatter.set_alpha(1.0)

            # Callout labels
            callout_a = float(ease_out(np.clip((progress - 0.1) / 0.3, 0, 1)))
            for lbl in self.callout_labels:
                lbl.set_alpha(callout_a * 0.8)

            # Astoria stat
            self.astoria_label.set_alpha(
                float(ease_out(np.clip((progress - 0.15) / 0.3, 0, 1))))

            # Labels
            for lbl in self.node_labels:
                if lbl:
                    lbl.set_alpha(0.7)

        # ── Outro ──
        elif phase == 'outro':
            # Keep act2 flowing but fade gradually
            fade = float(1.0 - ease_out(np.clip(progress * 2, 0, 1)))
            self._advance_particles(self.p2_t, self.p2_speed)
            pos2 = self._compute_positions(self.p2_t, self.p2_edge, self.act2_paths)
            self.p2_scatter.set_offsets(pos2)
            p_alpha2 = particle_alpha(self.p2_t) * max(fade, 0.3)
            colors2 = self.p2_base_colors.copy()
            colors2[:, 3] = p_alpha2
            self.p2_scatter.set_facecolors(colors2)

            # Keep map elements
            for line in self.act2_lines:
                line.set_alpha(0.25 * max(fade, 0.4))
            self.astoria_scatter.set_alpha(max(fade, 0.5))
            glow_size = 500 + 100 * np.sin(frame * 0.1)
            self.astoria_glow.set_sizes([glow_size])
            self.astoria_glow.set_alpha(max(fade, 0.3) * 0.4)

            # Fade callout labels
            for lbl in self.callout_labels:
                lbl.set_alpha(0.5 * max(fade, 0.3))
            self.astoria_label.set_alpha(max(fade, 0.3))

            # Outro stats panel fades in
            stats_a = float(ease_out(np.clip((progress - 0.1) / 0.4, 0, 1)))
            for t in self.outro_texts:
                t.set_alpha(stats_a)

            # Credit
            credit_a = float(ease_out(np.clip((progress - 0.3) / 0.4, 0, 1)))
            self.credit_text.set_alpha(credit_a)

        return []

    # ── Render ────────────────────────────────────────────────────────────

    def render(self):
        print("Creating animation...")
        anim = FuncAnimation(
            self.fig, self._update,
            frames=TOTAL_FRAMES,
            interval=1000 // FPS,
            blit=False)

        # Try MP4 via ffmpeg
        saved_mp4 = False
        try:
            # Try imageio-ffmpeg first
            try:
                import imageio_ffmpeg
                plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError:
                pass

            writer = FFMpegWriter(fps=FPS, bitrate=5000,
                                  extra_args=['-pix_fmt', 'yuv420p'])
            anim.save('transit_animation.mp4', writer=writer,
                      savefig_kwargs={'facecolor': BG_COLOR})
            print("Saved transit_animation.mp4")
            saved_mp4 = True
        except Exception as e:
            print(f"FFMpeg failed: {e}")
            print("Falling back to GIF (reduced resolution)...")
            # Reduce for GIF
            self.fig.set_size_inches(9.6, 5.4)
            self.fig.set_dpi(100)
            writer = PillowWriter(fps=15)
            anim.save('transit_animation.gif', writer=writer,
                      savefig_kwargs={'facecolor': BG_COLOR})
            print("Saved transit_animation.gif")

        # Save keyframe (proposed state, mid-act2)
        if saved_mp4:
            # Reset figure size if changed
            self.fig.set_size_inches(19.2, 10.8)
            self.fig.set_dpi(100)

        keyframe = 630  # mid Act 2
        self._update(keyframe)
        self.fig.savefig('transit_keyframe.png', dpi=100,
                         facecolor=BG_COLOR)
        print("Saved transit_keyframe.png")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Building transit flow animation...")
    ta = TransitAnimation()
    ta.render()
    print("Done.")
