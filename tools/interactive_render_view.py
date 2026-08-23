"""
Interactive Open3D viewer for myno2paper point-cloud visualizations.

Use the mouse to adjust the camera, then export the current view as PNG images.
The script applies the same reference normalization to all loaded PLY files, so
RGB, GT, baseline, and proposed results for one plant can share one camera.
"""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

from render_fixed_view import (
    DEFAULT_FILES,
    DEFAULT_MAX_POINTS,
    apply_transform,
    collect_files,
    compute_reference_transform,
    load_pcd,
    parse_background,
)

SEMANTIC_COLORS = np.array(
    [
        [88, 181, 74],
        [240, 194, 73],
        [155, 94, 171],
    ],
    dtype=np.float64,
) / 255.0
UNASSIGNED_INSTANCE_COLORS = np.array(
    [
        [80, 80, 80],
        [180, 180, 180],
    ],
    dtype=np.float64,
) / 255.0


HELP_TEXT = """
Interactive controls:
  Mouse         rotate / pan / zoom
  N            next PLY
  P            previous PLY
  S            save current view of the current PLY
  A            export all loaded PLY files with the current camera
  V            save current camera parameters as JSON
  L            load camera parameters from JSON
  Panel        toggle gray-instance fill and adjust max distance
  Panel        match pred_instance colors to gt_instance colors
  Panel        adjust preview/capture point size
  H            print this help text
  Esc / Q      close the window
  Panel        open another scene folder / change output folder
"""


def choose_directory(title, initialdir=None, mustexist=True, allow_cancel=False, fallback=True):
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(
            title=title,
            initialdir=str(initialdir) if initialdir else None,
            mustexist=mustexist,
        )
        root.destroy()
        if selected:
            return Path(selected)
        if allow_cancel:
            return None
    except Exception as exc:
        print(f"Directory dialog unavailable ({exc}); falling back to terminal input.")
        if not fallback:
            return None

    prompt = f"{title}: "
    while True:
        value = input(prompt).strip().strip("\"'")
        if value:
            path = Path(value).expanduser()
            if not mustexist or path.is_dir():
                return path
            print(f"Directory does not exist: {path}")


def choose_file(title, initialdir=None, filetypes=None, allow_cancel=True, fallback=True):
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askopenfilename(
            title=title,
            initialdir=str(initialdir) if initialdir else None,
            filetypes=filetypes or (("PLY files", "*.ply"), ("All files", "*.*")),
        )
        root.destroy()
        if selected:
            return Path(selected)
        if allow_cancel:
            return None
    except Exception as exc:
        print(f"File dialog unavailable ({exc}); falling back to terminal input.")
        if not fallback:
            return None

    value = input(f"{title} (leave empty to skip): ").strip().strip("\"'")
    return Path(value).expanduser() if value else None


def ask_yes_no(title, message, default=False):
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        result = messagebox.askyesno(title, message, default="yes" if default else "no")
        root.destroy()
        return result
    except Exception as exc:
        print(f"Question dialog unavailable ({exc}); falling back to terminal input.")
        suffix = "[Y/n]" if default else "[y/N]"
        value = input(f"{message} {suffix} ").strip().lower()
        if not value:
            return default
        return value in {"y", "yes"}


def parse_capture_point_scale(value):
    if isinstance(value, str) and value.strip().lower() == "auto":
        return "auto"
    scale = float(value)
    if scale <= 0:
        raise argparse.ArgumentTypeError("capture point scale must be positive or 'auto'")
    return scale


def resolve_interactive_paths(args):
    if args.scene_dir is None or args.select_paths:
        args.scene_dir = choose_directory(
            "Select input scene directory containing PLY files",
            initialdir=Path.cwd(),
            mustexist=True,
        )
    else:
        args.scene_dir = Path(args.scene_dir)

    if args.out_dir is None or args.select_paths:
        default_parent = args.scene_dir.parent if args.scene_dir else Path.cwd()
        args.out_dir = choose_directory(
            "Select output directory for exported PNG files",
            initialdir=default_parent,
            mustexist=False,
        )
    else:
        args.out_dir = Path(args.out_dir)

    if args.select_ref_ply:
        args.ref_ply = choose_file(
            "Select optional reference PLY for shared camera normalization",
            initialdir=args.scene_dir,
            filetypes=(("PLY files", "*.ply"), ("All files", "*.*")),
        )
    elif args.ref_ply:
        args.ref_ply = Path(args.ref_ply)
    return args


class InteractiveRenderer:
    def __init__(
        self,
        scene_dir,
        out_dir,
        ref_ply=None,
        files=None,
        align="pca-xy",
        yaw_deg=0.0,
        max_points=DEFAULT_MAX_POINTS,
        seed=20240611,
        width=1400,
        height=1000,
        point_size=2.0,
        capture_point_scale="auto",
        background=(1.0, 1.0, 1.0),
        camera_json=None,
        no_control_panel=False,
        fill_unassigned_instance=False,
        fill_max_distance=0.01,
        match_instance_colors=False,
    ):
        self.scene_dir = Path(scene_dir)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        self.point_size = point_size
        self.capture_point_scale = capture_point_scale
        self.background = np.array(background, dtype=np.float64)
        self.camera_json = Path(camera_json) if camera_json else None
        self.no_control_panel = no_control_panel
        self.control_root = None
        self.status_var = None
        self.file_listbox = None
        self._updating_listbox = False
        self.align = align
        self.yaw_deg = yaw_deg
        self.max_points = max_points
        self.seed = seed
        self.files = files
        self.fill_unassigned_instance = fill_unassigned_instance
        self.fill_max_distance = fill_max_distance
        self.match_instance_colors = match_instance_colors
        self.fill_var = None
        self.fill_distance_var = None
        self.match_var = None
        self.point_size_var = None
        self.capture_point_scale_var = None

        self.load_scene(self.scene_dir, ref_ply=ref_ply, files=files)

        self.index = 0
        self.vis = o3d.visualization.VisualizerWithKeyCallback()

    @staticmethod
    def colors_to_uint8(colors):
        colors = np.asarray(colors)
        if colors.size == 0:
            return colors.reshape(-1, 3).astype(np.uint8)
        if colors.max() <= 1.5:
            colors = colors * 255.0
        return np.clip(np.rint(colors), 0, 255).astype(np.uint8)

    @staticmethod
    def semantic_labels_from_colors(colors):
        if colors.size == 0:
            return np.zeros(0, dtype=np.int64)
        diff = colors[:, None, :] - SEMANTIC_COLORS[None, :, :]
        return np.argmin(np.sum(diff * diff, axis=2), axis=1).astype(np.int64)

    @staticmethod
    def unassigned_mask_from_colors(colors):
        if colors.size == 0:
            return np.zeros(0, dtype=bool)
        diff = colors[:, None, :] - UNASSIGNED_INSTANCE_COLORS[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        return np.min(dist2, axis=1) < 1e-8

    def unassigned_mask_from_uint8(self, colors):
        if colors.size == 0:
            return np.zeros(0, dtype=bool)
        gray = self.colors_to_uint8(UNASSIGNED_INSTANCE_COLORS)
        diff = colors[:, None, :].astype(np.int16) - gray[None, :, :].astype(np.int16)
        return np.min(np.sum(diff * diff, axis=2), axis=1) == 0

    def fill_instance_colors(self, raw_pcds):
        if not self.fill_unassigned_instance:
            return
        if "pred_semantic" not in raw_pcds or "pred_instance" not in raw_pcds:
            print("Skip fill: pred_semantic.ply or pred_instance.ply is missing.")
            return

        sem_pcd = raw_pcds["pred_semantic"]
        inst_pcd = raw_pcds["pred_instance"]
        sem_colors = np.asarray(sem_pcd.colors)
        inst_colors = np.asarray(inst_pcd.colors).copy()
        inst_points = np.asarray(inst_pcd.points)
        if sem_colors.shape[0] != inst_colors.shape[0]:
            print("Skip fill: pred_semantic and pred_instance point counts differ.")
            return

        labels = self.semantic_labels_from_colors(sem_colors)
        unassigned = self.unassigned_mask_from_colors(inst_colors)
        fill_count = 0
        for class_id in range(len(SEMANTIC_COLORS)):
            query_idx = np.where(unassigned & (labels == class_id))[0]
            assigned_idx = np.where((~unassigned) & (labels == class_id))[0]
            if query_idx.size == 0 or assigned_idx.size == 0:
                continue

            assigned_pcd = o3d.geometry.PointCloud()
            assigned_pcd.points = o3d.utility.Vector3dVector(inst_points[assigned_idx])
            tree = o3d.geometry.KDTreeFlann(assigned_pcd)

            for idx in query_idx:
                _, nn_idx, nn_dist2 = tree.search_knn_vector_3d(inst_points[idx], 1)
                if not nn_idx:
                    continue
                if self.fill_max_distance and self.fill_max_distance > 0:
                    if nn_dist2[0] ** 0.5 > self.fill_max_distance:
                        continue
                inst_colors[idx] = inst_colors[assigned_idx[nn_idx[0]]]
                fill_count += 1

        inst_pcd.colors = o3d.utility.Vector3dVector(inst_colors)
        print(f"Filled unassigned pred_instance points for visualization: {fill_count}")

    def match_pred_instance_colors_to_gt(self, raw_pcds):
        if not self.match_instance_colors:
            return
        if "gt_instance" not in raw_pcds or "pred_instance" not in raw_pcds:
            print("Skip color match: gt_instance.ply or pred_instance.ply is missing.")
            return

        gt_pcd = raw_pcds["gt_instance"]
        pred_pcd = raw_pcds["pred_instance"]
        gt_colors = self.colors_to_uint8(np.asarray(gt_pcd.colors))
        pred_colors = self.colors_to_uint8(np.asarray(pred_pcd.colors))
        if gt_colors.shape[0] != pred_colors.shape[0]:
            print("Skip color match: gt_instance and pred_instance point counts differ.")
            return

        gt_unassigned = self.unassigned_mask_from_uint8(gt_colors)
        pred_unassigned = self.unassigned_mask_from_uint8(pred_colors)
        pred_keys = np.ascontiguousarray(pred_colors).view(
            np.dtype((np.void, pred_colors.dtype.itemsize * pred_colors.shape[1]))
        ).reshape(-1)

        matched_count = 0
        out_colors = pred_colors.copy()
        for pred_key in np.unique(pred_keys[~pred_unassigned]):
            pred_mask = pred_keys == pred_key
            valid_gt = pred_mask & (~gt_unassigned)
            if not bool(valid_gt.any()):
                continue

            overlap_gt_colors, overlap_counts = np.unique(
                gt_colors[valid_gt], axis=0, return_counts=True
            )
            best_color = overlap_gt_colors[int(np.argmax(overlap_counts))]
            out_colors[pred_mask] = best_color
            matched_count += 1

        pred_pcd.colors = o3d.utility.Vector3dVector(out_colors.astype(np.float64) / 255.0)
        print(f"Matched pred_instance colors to GT instances: {matched_count}")

    def load_scene(self, scene_dir, ref_ply=None, files=None):
        self.scene_dir = Path(scene_dir)
        ref_path = Path(ref_ply) if ref_ply else self.scene_dir / "rgb.ply"
        ref_pcd = load_pcd(ref_path)
        center, rotation, scale = compute_reference_transform(
            ref_pcd, align=self.align, yaw_deg=self.yaw_deg
        )

        self.names = []
        self.pcds = []
        paths = collect_files(self.scene_dir, files)
        raw_pcds = {path.stem: load_pcd(path) for path in paths}
        self.fill_instance_colors(raw_pcds)
        self.match_pred_instance_colors_to_gt(raw_pcds)

        for path in paths:
            pcd = raw_pcds[path.stem]
            pcd = apply_transform(
                pcd,
                center=center,
                rotation=rotation,
                scale=scale,
                max_points=self.max_points,
                seed=self.seed,
                source_name=path.name,
            )
            self.names.append(path.stem)
            self.pcds.append(pcd)

        if not self.pcds:
            raise RuntimeError(f"No PLY files loaded from: {self.scene_dir}")
        self.ref_ply = ref_path
        self.files = files

    def setup_render_options(self, point_size=None):
        opt = self.vis.get_render_option()
        opt.background_color = self.background
        opt.point_size = self.point_size if point_size is None else point_size
        opt.light_on = False

    def run(self):
        print(HELP_TEXT)
        print("Loaded files:")
        for name in self.names:
            print(f"  - {name}")

        self.vis.create_window(
            window_name=f"Interactive render: {self.scene_dir.name}",
            width=self.width,
            height=self.height,
            visible=True,
        )
        self.setup_render_options()
        self.vis.add_geometry(self.pcds[self.index])
        self.register_callbacks()
        if not self.no_control_panel:
            self.create_control_panel()
            self.vis.register_animation_callback(self.pump_control_panel)

        if self.camera_json and self.camera_json.exists():
            self.load_camera(self.camera_json)

        self.update_title()
        self.vis.run()
        self.destroy_control_panel()
        self.vis.destroy_window()

    def register_callbacks(self):
        self.vis.register_key_callback(ord("N"), lambda vis: self.switch(1))
        self.vis.register_key_callback(ord("P"), lambda vis: self.switch(-1))
        self.vis.register_key_callback(ord("S"), lambda vis: self.save_current())
        self.vis.register_key_callback(ord("A"), lambda vis: self.export_all())
        self.vis.register_key_callback(ord("V"), lambda vis: self.save_camera())
        self.vis.register_key_callback(ord("L"), lambda vis: self.load_camera_dialog())
        self.vis.register_key_callback(ord("H"), lambda vis: self.print_help())
        self.vis.register_key_callback(ord("Q"), lambda vis: self.close())

    def create_control_panel(self):
        try:
            import tkinter as tk
            from tkinter import ttk
        except Exception as exc:
            print(f"Control panel unavailable ({exc}); using Open3D keys only.")
            return

        root = tk.Tk()
        root.title("Open3D render controls")
        root.geometry("+80+80")
        root.attributes("-topmost", True)

        frame = ttk.Frame(root, padding=10)
        frame.grid(row=0, column=0, sticky="nsew")

        guide = (
            "Mouse: rotate / pan / zoom\n"
            "N / P: next / previous PLY\n"
            "S: save current PNG\n"
            "A: export all with current camera\n"
            "V: save camera JSON\n"
            "L: load camera JSON\n"
            "Panel: fill gray instance points\n"
            "Panel: match pred/GT instance colors\n"
            "Panel: preview/capture point size\n"
            "Panel: open scene / change output\n"
            "Q or Esc: close"
        )
        ttk.Label(frame, text="Key Guide", font=("", 11, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        ttk.Label(frame, text=guide, justify="left").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(4, 10)
        )

        ttk.Button(frame, text="Open Scene Folder", command=self.open_scene_dialog).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(0, 4)
        )
        ttk.Button(frame, text="Change Output Folder", command=self.change_output_dialog).grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=(0, 8)
        )

        fill_frame = ttk.LabelFrame(frame, text="Instance Display", padding=6)
        fill_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        self.fill_var = tk.BooleanVar(value=self.fill_unassigned_instance)
        ttk.Checkbutton(
            fill_frame,
            text="Fill gray instance points",
            variable=self.fill_var,
            command=self.apply_fill_settings,
        ).grid(row=0, column=0, columnspan=3, sticky="w")
        self.match_var = tk.BooleanVar(value=self.match_instance_colors)
        ttk.Checkbutton(
            fill_frame,
            text="Match pred colors to GT",
            variable=self.match_var,
            command=self.apply_fill_settings,
        ).grid(row=1, column=0, columnspan=3, sticky="w")
        ttk.Label(fill_frame, text="Max dist").grid(row=2, column=0, sticky="w")
        self.fill_distance_var = tk.StringVar(value=str(self.fill_max_distance))
        ttk.Entry(fill_frame, textvariable=self.fill_distance_var, width=8).grid(
            row=2, column=1, sticky="ew", padx=(4, 4)
        )
        ttk.Button(fill_frame, text="Apply", command=self.apply_fill_settings).grid(
            row=2, column=2, sticky="ew"
        )

        render_frame = ttk.LabelFrame(frame, text="Render Display", padding=6)
        render_frame.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        ttk.Label(render_frame, text="Point size").grid(row=0, column=0, sticky="w")
        self.point_size_var = tk.StringVar(value=str(self.point_size))
        ttk.Entry(render_frame, textvariable=self.point_size_var, width=8).grid(
            row=0, column=1, sticky="ew", padx=(4, 4)
        )
        ttk.Label(render_frame, text="Capture scale").grid(row=1, column=0, sticky="w")
        self.capture_point_scale_var = tk.StringVar(value=str(self.capture_point_scale))
        ttk.Entry(render_frame, textvariable=self.capture_point_scale_var, width=8).grid(
            row=1, column=1, sticky="ew", padx=(4, 4)
        )
        ttk.Button(render_frame, text="Apply", command=self.apply_render_settings).grid(
            row=0, column=2, rowspan=2, sticky="nsew"
        )
        render_frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Preview PLY", font=("", 11, "bold")).grid(
            row=6, column=0, columnspan=2, sticky="w", pady=(0, 4)
        )
        list_height = min(max(len(self.names), 3), 8)
        listbox = tk.Listbox(frame, height=list_height, exportselection=False)
        for name in self.names:
            listbox.insert(tk.END, name)
        listbox.selection_set(self.index)
        listbox.activate(self.index)
        listbox.grid(row=7, column=0, columnspan=2, sticky="nsew", pady=(0, 10))
        listbox.bind("<<ListboxSelect>>", self.on_listbox_select)
        self.file_listbox = listbox

        ttk.Button(frame, text="Previous (P)", command=lambda: self.switch(-1)).grid(
            row=8, column=0, sticky="ew", padx=(0, 4), pady=2
        )
        ttk.Button(frame, text="Next (N)", command=lambda: self.switch(1)).grid(
            row=8, column=1, sticky="ew", padx=(4, 0), pady=2
        )
        ttk.Button(frame, text="Save Current (S)", command=self.save_current).grid(
            row=9, column=0, sticky="ew", padx=(0, 4), pady=2
        )
        ttk.Button(frame, text="Export All (A)", command=self.export_all).grid(
            row=9, column=1, sticky="ew", padx=(4, 0), pady=2
        )
        ttk.Button(frame, text="Save Camera (V)", command=self.save_camera).grid(
            row=10, column=0, sticky="ew", padx=(0, 4), pady=2
        )
        ttk.Button(frame, text="Load Camera (L)", command=self.load_camera_dialog).grid(
            row=10, column=1, sticky="ew", padx=(4, 0), pady=2
        )
        ttk.Button(frame, text="Close (Q)", command=self.close).grid(
            row=11, column=0, columnspan=2, sticky="ew", pady=(8, 2)
        )

        self.status_var = tk.StringVar()
        ttk.Label(frame, textvariable=self.status_var, foreground="#333333").grid(
            row=12, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        root.protocol("WM_DELETE_WINDOW", self.close)

        self.control_root = root
        self.update_status()

    def on_listbox_select(self, event):
        del event
        if self._updating_listbox or self.file_listbox is None:
            return
        selection = self.file_listbox.curselection()
        if not selection:
            return
        self.switch_to_index(int(selection[0]))

    def pump_control_panel(self, vis):
        del vis
        if self.control_root is None:
            return False
        try:
            self.control_root.update_idletasks()
            self.control_root.update()
        except Exception:
            self.control_root = None
        return False

    def destroy_control_panel(self):
        if self.control_root is None:
            return
        try:
            self.control_root.destroy()
        except Exception:
            pass
        self.control_root = None

    def print_help(self):
        print(HELP_TEXT)
        return False

    def close(self):
        self.destroy_control_panel()
        self.vis.close()
        return False

    def update_title(self):
        print(f"Showing [{self.index + 1}/{len(self.pcds)}]: {self.names[self.index]}")
        self.update_status()

    def update_status(self):
        if self.status_var is not None:
            self.status_var.set(
                f"Scene: {self.scene_dir.name}\n"
                f"Output: {self.out_dir}\n"
                f"Fill gray instance: {self.fill_unassigned_instance}, "
                f"max dist: {self.fill_max_distance}\n"
                f"Match pred/GT colors: {self.match_instance_colors}\n"
                f"Point size: {self.point_size}, "
                f"capture scale: {self.capture_point_scale}\n"
                f"Showing {self.index + 1}/{len(self.pcds)}: {self.names[self.index]}"
            )
        if self.file_listbox is not None:
            self._updating_listbox = True
            try:
                self.file_listbox.selection_clear(0, "end")
                self.file_listbox.selection_set(self.index)
                self.file_listbox.activate(self.index)
                self.file_listbox.see(self.index)
            finally:
                self._updating_listbox = False

    def apply_fill_settings(self):
        if self.fill_var is not None:
            self.fill_unassigned_instance = bool(self.fill_var.get())
        if self.match_var is not None:
            self.match_instance_colors = bool(self.match_var.get())
        if self.fill_distance_var is not None:
            try:
                self.fill_max_distance = float(self.fill_distance_var.get())
            except ValueError:
                print(
                    f"Invalid fill distance: {self.fill_distance_var.get()}, "
                    f"keep {self.fill_max_distance}"
                )
        print(
            "Reload scene with fill settings: "
            f"fill={self.fill_unassigned_instance}, "
            f"max_distance={self.fill_max_distance}, "
            f"match_colors={self.match_instance_colors}"
        )
        return self.replace_scene(self.scene_dir, ref_ply=self.ref_ply)

    def apply_render_settings(self):
        if self.point_size_var is not None:
            try:
                point_size = float(self.point_size_var.get())
                if point_size <= 0:
                    raise ValueError
                self.point_size = point_size
            except ValueError:
                print(f"Invalid point size: {self.point_size_var.get()}, keep {self.point_size}")
                self.point_size_var.set(str(self.point_size))
        if self.capture_point_scale_var is not None:
            try:
                self.capture_point_scale = parse_capture_point_scale(
                    self.capture_point_scale_var.get()
                )
            except (ValueError, argparse.ArgumentTypeError):
                print(
                    "Invalid capture point scale: "
                    f"{self.capture_point_scale_var.get()}, keep {self.capture_point_scale}"
                )
                self.capture_point_scale_var.set(str(self.capture_point_scale))

        self.setup_render_options()
        self.vis.poll_events()
        self.vis.update_renderer()
        self.update_status()
        print(
            "Applied render settings: "
            f"point_size={self.point_size}, "
            f"capture_point_scale={self.capture_point_scale}"
        )
        return False

    def effective_capture_point_scale(self):
        if self.capture_point_scale != "auto":
            return float(self.capture_point_scale)
        try:
            self.vis.poll_events()
            self.vis.update_renderer()
            image = self.vis.capture_screen_float_buffer(do_render=False)
            array = np.asarray(image)
            if array.ndim >= 2 and array.shape[0] > 0 and array.shape[1] > 0:
                width_scale = array.shape[1] / max(float(self.width), 1.0)
                height_scale = array.shape[0] / max(float(self.height), 1.0)
                return max(width_scale, height_scale, 1.0)
        except Exception as exc:
            print(f"Could not estimate capture point scale ({exc}); use 1.0")
        return 1.0

    def rebuild_listbox(self):
        if self.file_listbox is None:
            return
        self._updating_listbox = True
        try:
            self.file_listbox.delete(0, "end")
            for name in self.names:
                self.file_listbox.insert("end", name)
            self.file_listbox.selection_set(self.index)
            self.file_listbox.activate(self.index)
            self.file_listbox.see(self.index)
        finally:
            self._updating_listbox = False

    def switch(self, step):
        return self.switch_to_index((self.index + step) % len(self.pcds))

    def switch_to_index(self, index):
        if index == self.index:
            self.update_title()
            return False
        if index < 0 or index >= len(self.pcds):
            print(f"Invalid PLY index: {index}")
            return False
        params = self.current_camera()
        self.vis.clear_geometries()
        self.index = index
        self.vis.add_geometry(self.pcds[self.index])
        self.setup_render_options()
        self.restore_camera(params)
        self.update_title()
        return False

    def replace_scene(self, scene_dir, ref_ply=None):
        params = self.current_camera()
        old_scene = self.scene_dir
        try:
            self.load_scene(scene_dir, ref_ply=ref_ply, files=self.files)
            self.index = 0
            self.vis.clear_geometries()
            self.vis.add_geometry(self.pcds[self.index])
            self.setup_render_options()
            self.restore_camera(params)
            self.rebuild_listbox()
            self.update_title()
            print(f"Loaded new scene: {self.scene_dir}")
        except Exception as exc:
            print(f"Failed to load scene {scene_dir}: {exc}", file=sys.stderr)
            if old_scene != self.scene_dir:
                try:
                    self.load_scene(old_scene, ref_ply=self.ref_ply, files=self.files)
                except Exception:
                    pass
        return False

    def open_scene_dialog(self):
        scene_dir = choose_directory(
            "Select another input scene directory containing PLY files",
            initialdir=self.scene_dir.parent,
            mustexist=True,
            allow_cancel=True,
            fallback=False,
        )
        if scene_dir is None:
            print("Open scene cancelled.")
            return False

        ref_ply = None
        if ask_yes_no(
            "Reference PLY",
            "Use a separate reference PLY for this scene?",
            default=False,
        ):
            ref_ply = choose_file(
                "Select reference PLY for shared normalization",
                initialdir=scene_dir,
                filetypes=(("PLY files", "*.ply"), ("All files", "*.*")),
                allow_cancel=True,
                fallback=False,
            )
        return self.replace_scene(scene_dir, ref_ply=ref_ply)

    def change_output_dialog(self):
        out_dir = choose_directory(
            "Select new output directory for exported PNG files",
            initialdir=self.out_dir,
            mustexist=False,
            allow_cancel=True,
            fallback=False,
        )
        if out_dir is None:
            print("Change output cancelled.")
            return False
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Changed output directory: {self.out_dir}")
        self.update_status()
        return False

    def current_camera(self):
        return copy.deepcopy(
            self.vis.get_view_control().convert_to_pinhole_camera_parameters()
        )

    def restore_camera(self, params):
        if params is None:
            return
        self.vis.poll_events()
        self.vis.update_renderer()
        self.vis.get_view_control().convert_from_pinhole_camera_parameters(
            params, allow_arbitrary=True
        )
        self.vis.poll_events()
        self.vis.update_renderer()

    def capture_current_image(self, out_path):
        params = self.current_camera()
        effective_scale = self.effective_capture_point_scale()
        capture_point_size = self.point_size * effective_scale
        self.setup_render_options(point_size=capture_point_size)
        self.restore_camera(params)
        self.vis.poll_events()
        self.vis.update_renderer()
        self.vis.capture_screen_image(str(out_path), do_render=True)
        self.setup_render_options()
        self.restore_camera(params)
        print(
            f"Saved: {out_path} "
            f"(point_size={self.point_size}, "
            f"capture_scale={self.capture_point_scale}, effective={effective_scale:.3f})"
        )

    def save_current(self):
        out_path = self.out_dir / f"{self.names[self.index]}_interactive.png"
        self.capture_current_image(out_path)
        return False

    def export_all(self):
        params = self.current_camera()
        active_index = self.index
        for idx, (name, pcd) in enumerate(zip(self.names, self.pcds)):
            self.vis.clear_geometries()
            self.vis.add_geometry(pcd)
            self.setup_render_options()
            self.restore_camera(params)
            out_path = self.out_dir / f"{name}_interactive.png"
            self.capture_current_image(out_path)
            self.index = idx

        self.vis.clear_geometries()
        self.index = active_index
        self.vis.add_geometry(self.pcds[self.index])
        self.setup_render_options()
        self.restore_camera(params)
        self.update_title()
        return False

    def save_camera(self):
        out_path = self.out_dir / "camera_interactive.json"
        params = self.current_camera()
        o3d.io.write_pinhole_camera_parameters(str(out_path), params)
        print(f"Saved camera parameters: {out_path}")
        return False

    def load_camera(self, path):
        params = o3d.io.read_pinhole_camera_parameters(str(path))
        self.restore_camera(params)
        print(f"Loaded camera parameters: {path}")
        return False

    def load_camera_dialog(self):
        path = choose_file(
            "Select camera JSON to load",
            initialdir=self.out_dir,
            filetypes=(("Camera JSON", "*.json"), ("All files", "*.*")),
        )
        if path is None:
            print("Camera load cancelled.")
            return False
        if not path.exists():
            print(f"Camera JSON does not exist: {path}")
            return False
        return self.load_camera(path)


def main():
    parser = argparse.ArgumentParser(
        description="Interactive fixed-reference Open3D viewer with PNG export."
    )
    parser.add_argument(
        "--scene-dir",
        default=None,
        help="Directory containing PLY files. If omitted, a directory chooser opens.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory for PNG/camera outputs. If omitted, a directory chooser opens.",
    )
    parser.add_argument(
        "--select-paths",
        action="store_true",
        help="Open directory choosers even when --scene-dir/--out-dir are provided.",
    )
    parser.add_argument(
        "--ref-ply",
        default=None,
        help="Reference PLY used for centering/scaling/PCA. Defaults to scene-dir/rgb.ply.",
    )
    parser.add_argument(
        "--select-ref-ply",
        action="store_true",
        help="Open a file chooser for the optional reference PLY.",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help=f"PLY filenames under scene-dir. Defaults to: {', '.join(DEFAULT_FILES)}",
    )
    parser.add_argument(
        "--align",
        choices=("pca-xy", "none"),
        default="pca-xy",
        help="How to align the plant before rendering.",
    )
    parser.add_argument("--yaw-deg", type=float, default=0.0)
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument(
        "--capture-point-scale",
        type=parse_capture_point_scale,
        default="auto",
        help=(
            "Temporary multiplier applied to point size only while saving PNGs. "
            "Use 'auto' to compensate display/capture DPI differences."
        ),
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=DEFAULT_MAX_POINTS,
        help="Maximum loaded points per PLY. Use 0 to disable downsampling.",
    )
    parser.add_argument("--seed", type=int, default=20240611)
    parser.add_argument(
        "--background",
        type=parse_background,
        default=(1.0, 1.0, 1.0),
        help="white, black, gray, or comma-separated RGB values in [0,1].",
    )
    parser.add_argument(
        "--camera-json",
        default=None,
        help="Optional camera JSON previously saved with V.",
    )
    parser.add_argument(
        "--no-control-panel",
        action="store_true",
        help="Disable the Tk key-guide/control panel.",
    )
    parser.add_argument(
        "--fill-unassigned-instance",
        action="store_true",
        help=(
            "Visualization only: fill gray pred_instance points from nearest "
            "same-semantic instance color."
        ),
    )
    parser.add_argument(
        "--fill-max-distance",
        type=float,
        default=0.01,
        help="Maximum raw-coordinate distance for --fill-unassigned-instance. 0 disables the limit.",
    )
    parser.add_argument(
        "--match-instance-colors",
        action="store_true",
        help=(
            "Visualization only: recolor pred_instance by majority-overlap "
            "gt_instance colors."
        ),
    )
    args = parser.parse_args()
    args = resolve_interactive_paths(args)

    try:
        renderer_args = vars(args)
        renderer_args.pop("select_paths")
        renderer_args.pop("select_ref_ply")
        renderer = InteractiveRenderer(**renderer_args)
        renderer.run()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
