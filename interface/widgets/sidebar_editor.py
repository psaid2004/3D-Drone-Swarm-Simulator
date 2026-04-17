#TODO: Add search function
#TODO: Expand/collapse for objects and targets as well, not just swarms
#TODO: Pre play simulation in the background for replayability; 
from functools import partial

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from envs.createWorld import SUPPORTED_SHAPES
from interface.core.config_manager import (
    AXIS_INT_TO_LABEL,
    KINDS,
    SWARM_COLOR_PRESETS,
    ConfigManager,
)

SIDEBAR_MODES = ("object_spawns", "drone_spawns", "target_spawns")

# Color palette for objects - cycles through these colors
OBJECT_COLORS = [
    "#E6F5D4",  # Verde lima muy claro
    "#D9F0C3",  # Verde menta suave
    "#CCEBB2",  # Verde pastel
    "#BFE6A1",  # Verde manzana claro
    "#B2E190",  # Verde primavera
    "#A5DC7F",  # Verde salvia claro
    "#98D76E",  # Verde hierba suave
    "#8BD25D",  # Verde pistacho
    "#7ECD4C",  # Verde musgo claro
    "#C8EDB8",  # Verde té verde
    "#BEE8AB",  # Verde pera
    "#B4E39E",  # Verde albahaca clara
    "#AADE91",  # Verde esmeralda claro
    "#A0D984",  # Verde oliva claro
    "#96D477",  # Verde césped suave
    "#DCF5CC",  # Verde nieve
    "#D2F0BF",  # Verde junco
    "#F0FAD8",  # Verde muy pálido
    "#EAF9D0",  # Verde lima pálido
    "#E2F7C8",  # Verde brillante claro
]

class _NoScrollComboBox(QComboBox):
    def wheelEvent(self, event):
        event.ignore()


class SidebarEditor(QWidget):
    """
    Left-sidebar widget.
    Contains three modes for editing the object spawn configs, drone spawn configs, and target spawn configs.
    Allows the user to edit the fields of each config entry, and emits signals when changes are made that require the simulation to be restarted.
    """

    restart_requested = pyqtSignal(str)     # Emitted when a change is made that requires the simulation to be restarted.
    swarm_count_changed = pyqtSignal(int)   # Emitted when the swarm count is changed, so that the main window can update its drone count.
    randomize_requested = pyqtSignal()      # Emmited when the "Generate Random" button is clicked.

    def __init__(
            self, config_manager: ConfigManager,
            drones_per_swarm: int, 
            parent=None,
            max_swarms: int = 4,
            max_drones_per_swarm: int = 10,):
        super().__init__(parent)
        self._config_manager = config_manager
        self._drones_per_swarm = drones_per_swarm
        self._max_swarms = max_swarms
        self._max_drones_per_swarm = max_drones_per_swarm
        self._sidebar_mode = "object_spawns"
        self._expanded_swarms: set = set() # indicates which swarm configs have their drone details expanded in the UI
        self._expanded_objects: set = set() # indicates which object configs have their details expanded in the UI
        self._object_filter = "all"  # Filter for objects: "all", "static", or "dynamic"
        self._object_search = ""  # Search filter for object names
        self._build_ui()
        self._set_sidebar_mode("object_spawns")
        self._update_table = None

    # ---- UI construction ----

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 10)

        self.side_title = QLabel("Objects")
        self.side_title.setStyleSheet("font-weight: 700;")

        # Mode selection buttons
        mode_row = QWidget()
        mode_layout = QHBoxLayout(mode_row)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(4)

        self.mode_btn_objects = QPushButton("Obstacles")
        self.mode_btn_drones = QPushButton("Drones")
        self.mode_btn_targets = QPushButton("Targets")
        for btn in (self.mode_btn_objects, self.mode_btn_drones, self.mode_btn_targets):
            btn.setCheckable(True)
            btn.setStyleSheet(
                "QPushButton { border: 1px solid #ccc; border-radius: 3px; padding: 2px; }"
                "QPushButton:checked { background-color: #C8EDB8; border: 2px solid #7ECD4C; }"
            )
            mode_layout.addWidget(btn)
        self.mode_btn_objects.clicked.connect(lambda: self._set_sidebar_mode("object_spawns"))
        self.mode_btn_drones.clicked.connect(lambda: self._set_sidebar_mode("drone_spawns"))
        self.mode_btn_targets.clicked.connect(lambda: self._set_sidebar_mode("target_spawns"))

        # Count buttons
        self.count_label = QLabel("Object count")
        self.count_spinbox = QSpinBox()
        self.count_spinbox.setValue(self._config_manager.count("object_spawns")) 
        self.count_spinbox.valueChanged.connect(self._on_count_spinbox) 

        # Scroll area for config cards
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_host = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_host)
        self.scroll_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_layout.setSpacing(8)
        self.scroll_area.setWidget(self.scroll_host)

        layout.addWidget(self.side_title)
        layout.addWidget(mode_row)
        
        # Filter buttons for objects (shown only in object mode)
        self.filter_row = QWidget()
        filter_layout = QHBoxLayout(self.filter_row)
        filter_layout.setContentsMargins(0, 0, 0, 0)
        filter_layout.setSpacing(4)
        
        self.filter_btn_all = QPushButton("All")
        self.filter_btn_static = QPushButton("Static")
        self.filter_btn_dynamic = QPushButton("Dynamic")
        for btn in (self.filter_btn_all, self.filter_btn_static, self.filter_btn_dynamic):
            btn.setCheckable(True)
            btn.setStyleSheet(
                "QPushButton { border: 1px solid #ccc; border-radius: 3px; padding: 2px; }"
                "QPushButton:checked { background-color: #C8EDB8; border: 2px solid #7ECD4C; }"
            )
            filter_layout.addWidget(btn)
        
        self.filter_btn_all.clicked.connect(lambda: self._set_object_filter("all"))
        self.filter_btn_static.clicked.connect(lambda: self._set_object_filter("static"))
        self.filter_btn_dynamic.clicked.connect(lambda: self._set_object_filter("dynamic"))
        
        self.filter_row.setVisible(False)  # Hidden by default, shown when in object mode
        layout.addWidget(self.filter_row)
        
        # Search box for objects (shown only in object mode)
        self.search_row = QWidget()
        search_layout = QHBoxLayout(self.search_row)
        search_layout.setContentsMargins(0, 0, 0, 0)
        search_layout.setSpacing(4)
        search_label = QLabel("Search:")
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Filter by name...")
        self.search_box.textChanged.connect(self._on_search_text_changed)
        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_box)
        
        self.search_row.setVisible(False)  # Hidden by default, shown when in object mode
        layout.addWidget(self.search_row)
        
        self.randomize_btn = QPushButton("Generate Random")
        self.randomize_btn.clicked.connect(lambda: self.randomize_requested.emit())

        layout.addWidget(self.count_label)
        layout.addWidget(self.count_spinbox)
        layout.addWidget(self.randomize_btn)
        layout.addWidget(self.scroll_area)

    # ---- Mode switching ----

    def _set_sidebar_mode(self, mode: str):
        """
        Switches the sidebar to show one of the three modes: object spawns, drone spawns, or target spawns.
        The mode buttons also update their checked state to reflect the current mode.
        """
        if mode not in SIDEBAR_MODES:
            return
        self._sidebar_mode = mode

        self.mode_btn_objects.setChecked(mode == "object_spawns")
        self.mode_btn_drones.setChecked(mode == "drone_spawns")
        self.mode_btn_targets.setChecked(mode == "target_spawns")

        # Show filter row and search row only in object mode
        self.filter_row.setVisible(mode == "object_spawns")
        self.search_row.setVisible(mode == "object_spawns")

        if mode == "object_spawns":
            self.side_title.setText("Objects")
            self.count_label.setText("Object count")
            self._object_filter = "all"
            self._object_search = ""
            self.search_box.blockSignals(True)
            self.search_box.setText("")
            self.search_box.blockSignals(False)
            self._update_filter_buttons()
            self.count_spinbox.setRange(0, 500)
        elif mode == "drone_spawns":
            self.side_title.setText("Swarms")
            self.count_label.setText("Swarm count")
        else:
            self.side_title.setText("Targets")
            self.count_label.setText("Target count")
        self.count_spinbox.blockSignals(True)
        if mode == "drone_spawns" or mode == "target_spawns":
            self.count_spinbox.setRange(0, self._max_swarms)
        self.count_spinbox.setValue(self._config_manager.count(mode))
        self.count_spinbox.blockSignals(False)
        self.refresh()

    # ---- Count spinbox ----

    def _on_count_spinbox(self, value: int):
        """
        Handler for when the count spinbox value is changed.
        Updates the config manager with the new count, refreshes the UI, and emits a restart request signal.
        """
        if self._sidebar_mode == "object_spawns":
            self._config_manager.sync_object_count(int(value))
            self.refresh()
            self.restart_requested.emit(f"Restarting with updated count: {value}")
        elif self._sidebar_mode == "drone_spawns":
            self._config_manager.sync_swarm_count(int(value))
            self.refresh()
            self.swarm_count_changed.emit(int(value))
            self.restart_requested.emit(f"Restarting with updated swarm count: {value}")
        else:
            self._config_manager.sync_target_count(int(value))
            self.refresh()
            self.swarm_count_changed.emit(int(value))
            self.restart_requested.emit(f"Restarting with updated count: {value}")

    # ---- Scroll area rebuild ----

    def refresh(self):
        """
        Rebuilds the scroll area with config cards based on the current sidebar mode and the configs in the config manager.
        """
        while self.scroll_layout.count():
            item = self.scroll_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        if self._sidebar_mode == "drone_spawns":
            self._build_drone_cards()
            return

        if self._sidebar_mode == "target_spawns":
            self._build_target_cards()
            return

        self._build_object_cards()

    def _build_drone_cards(self):
        """
        Builds the config cards for the drone spawn mode, which are organized by swarms. 
        Each swarm card can be expanded to show the individual drone configs within that swarm.
        """
        swarms = self._config_manager.get_swarm_configs()
        drones = self._config_manager.get_drone_spawn_configs()

        for swarm_idx, swarm_cfg in enumerate(swarms):
            swarm_name = swarm_cfg.get("name", f"Swarm {swarm_idx + 1}")
            card = QGroupBox("")
            
            # Apply color styling to the swarm card
            color = self._get_object_color(swarm_idx)
            card.setStyleSheet(
                f"QGroupBox {{"
                f"  background-color: {color};"
                f"  border: 2px solid {color};"
                f"  border-radius: 5px;"
                f"  margin-top: 0px;"
                f"  padding-top: 0px;"
                f"}}"
            )
            
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(8, 8, 8, 8)
            card_layout.setSpacing(6)
            
            # Add swarm name label
            name_label = QLabel(swarm_name)
            name_label.setStyleSheet("font-weight: 700; font-size: 11pt;")
            card_layout.addWidget(name_label)

            toggle_btn = QPushButton(
                "Collapse drones" if swarm_idx in self._expanded_swarms else "Expand drones"
            )
            toggle_btn.clicked.connect(partial(self._toggle_swarm_expand, swarm_idx))
            card_layout.addWidget(toggle_btn)

            card_layout.addWidget(self._labeled_entry(
                card, "Name", swarm_name,
                partial(self._on_swarm_field, swarm_idx, "name"),
            ))
            card_layout.addWidget(self._labeled_entry(
                card, "Center X", swarm_cfg.get("x", 0.0),
                partial(self._on_swarm_field, swarm_idx, "x"),
            ))
            card_layout.addWidget(self._labeled_entry(
                card, "Center Y", swarm_cfg.get("y", 0.0),
                partial(self._on_swarm_field, swarm_idx, "y"),
            ))
            card_layout.addWidget(self._labeled_entry(
                card, "Center Z", swarm_cfg.get("z", 0.8),
                partial(self._on_swarm_field, swarm_idx, "z"),
            ))
            card_layout.addWidget(self._option_menu(
                card, "Color",
                str(swarm_cfg.get("color_name", "Green")),
                tuple(SWARM_COLOR_PRESETS.keys()),
                partial(self._on_swarm_field, swarm_idx, "color_name"),
            ))

            if swarm_idx in self._expanded_swarms:
                for drone_local_idx in range(self._drones_per_swarm):
                    drone_index = swarm_idx * self._drones_per_swarm + drone_local_idx
                    if not (0 <= drone_index < len(drones)):
                        continue
                    drone_cfg = drones[drone_index]

                    drone_card = QGroupBox("")
                    
                    # Apply color styling to the drone card
                    drone_color = self._get_object_color(drone_index)
                    drone_card.setStyleSheet(
                        f"QGroupBox {{"
                        f"  background-color: {drone_color};"
                        f"  border: 2px solid {drone_color};"
                        f"  border-radius: 5px;"
                        f"  margin-top: 0px;"
                        f"  padding-top: 0px;"
                        f"}}"
                    )
                    
                    drone_layout = QVBoxLayout(drone_card)
                    drone_layout.setContentsMargins(8, 8, 8, 8)
                    drone_layout.setSpacing(4)
                    
                    # Add drone name label
                    drone_name_label = QLabel(f"Drone {drone_local_idx + 1}")
                    drone_name_label.setStyleSheet("font-weight: 700; font-size: 11pt;")
                    drone_layout.addWidget(drone_name_label)
                    drone_layout.addWidget(self._labeled_entry(
                        drone_card, "Offset X", drone_cfg.get("local_x", 0.0),
                        partial(self._on_drone_offset, drone_index, "local_x"),
                    ))
                    drone_layout.addWidget(self._labeled_entry(
                        drone_card, "Offset Y", drone_cfg.get("local_y", 0.0),
                        partial(self._on_drone_offset, drone_index, "local_y"),
                    ))
                    drone_layout.addWidget(self._labeled_entry(
                        drone_card, "Offset Z", drone_cfg.get("local_z", 0.0),
                        partial(self._on_drone_offset, drone_index, "local_z"),
                    ))
                    card_layout.addWidget(drone_card)

            self.scroll_layout.addWidget(card)
        self.scroll_layout.addStretch(1)

    def _build_target_cards(self):
        configs = self._config_manager.get_target_spawn_configs()
        if not configs:
            self._show_empty()
            return
        for i, raw_cfg in enumerate(configs):
            cfg = self._config_manager.ensure_spawn_fields(raw_cfg, "Target")
            target_name = cfg.get("name", f"Target {i + 1}")
            card = QGroupBox("")
            
            # Apply color styling to the target card
            color = self._get_object_color(i)
            card.setStyleSheet(
                f"QGroupBox {{"
                f"  background-color: {color};"
                f"  border: 2px solid {color};"
                f"  border-radius: 5px;"
                f"  margin-top: 0px;"
                f"  padding-top: 0px;"
                f"}}"
            )
            
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(8, 8, 8, 8)
            card_layout.setSpacing(4)
            
            # Add target name label
            name_label = QLabel(target_name)
            name_label.setStyleSheet("font-weight: 700; font-size: 11pt;")
            card_layout.addWidget(name_label)
            card_layout.addWidget(self._labeled_entry(
                card, "Name", target_name,
                partial(self._on_spawn_field, "target_spawns", i, "name"),
            ))
            card_layout.addWidget(self._labeled_entry(
                card, "Coord X", cfg["x"],
                partial(self._on_spawn_field, "target_spawns", i, "x"),
            ))
            card_layout.addWidget(self._labeled_entry(
                card, "Coord Y", cfg["y"],
                partial(self._on_spawn_field, "target_spawns", i, "y"),
            ))
            card_layout.addWidget(self._labeled_entry(
                card, "Coord Z", cfg["z"],
                partial(self._on_spawn_field, "target_spawns", i, "z"),
            ))
            self.scroll_layout.addWidget(card)
        self.scroll_layout.addStretch(1)

    def _build_object_cards(self):
        configs = self._config_manager.get_object_configs()
        if not configs:
            self._show_empty()
            return
        
        # Filter objects based on current filter setting and search text
        filtered_configs = []
        search_lower = self._object_search.lower()
        for i, raw_cfg in enumerate(configs):
            cfg = self._config_manager._ensure_object_fields(dict(raw_cfg))
            obj_kind = cfg.get("kind", "static")
            obj_name = cfg.get("name", f"Object {i + 1}").lower()
            
            # Apply type filter
            if self._object_filter == "static" and obj_kind != "static":
                continue
            elif self._object_filter == "dynamic" and obj_kind != "dynamic":
                continue
            
            # Apply search filter
            if search_lower and search_lower not in obj_name:
                continue
            
            filtered_configs.append((i, raw_cfg, cfg))
        
        if not filtered_configs:
            self._show_empty()
            return
        
        for i, raw_cfg, cfg in filtered_configs:
            obj_name = cfg.get("name", f"Object {i + 1}")
            card = QGroupBox("")
            
            # Apply color styling to the card
            color = self._get_object_color(i)
            card.setStyleSheet(
                f"QGroupBox {{"
                f"  background-color: {color};"
                f"  border: 2px solid {color};"
                f"  border-radius: 5px;"
                f"  margin-top: 0px;"
                f"  padding-top: 0px;"
                f"}}"
            )
            
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(8, 8, 8, 8)
            card_layout.setSpacing(4)

            # Add object name label in top left corner
            name_label = QLabel(obj_name)
            name_label.setStyleSheet("font-weight: 700; font-size: 11pt;")
            card_layout.addWidget(name_label)

            header_row = QWidget(card)
            header_layout = QHBoxLayout(header_row)
            header_layout.setContentsMargins(0, 0, 0, 0)
            header_layout.addStretch(1)
            header_layout.addWidget(self._remove_object_button(i), 0, Qt.AlignmentFlag.AlignRight)
            card_layout.addWidget(header_row)

            toggle_btn = QPushButton(
                "Collapse details" if i in self._expanded_objects else "Expand details"
            )
            toggle_btn.clicked.connect(partial(self._toggle_object_expand, i))
            card_layout.addWidget(toggle_btn)

            card_layout.addWidget(self._labeled_entry(
                card, "Name", cfg["name"],
                partial(self._on_object_field, i, "name"),
            ))
            card_layout.addWidget(self._option_menu(
                card, "Type", cfg["kind"], KINDS,
                partial(self._on_object_field, i, "kind"),
            ))

            if i in self._expanded_objects:
                card_layout.addWidget(self._option_menu(
                    card, "Shape", cfg["shape"], SUPPORTED_SHAPES,
                    partial(self._on_object_field, i, "shape"),
                ))
                card_layout.addWidget(self._labeled_entry(
                    card, "Coord X", cfg["x"],
                    partial(self._on_object_field, i, "x"),
                ))
                card_layout.addWidget(self._labeled_entry(
                    card, "Coord Y", cfg["y"],
                    partial(self._on_object_field, i, "y"),
                ))
                card_layout.addWidget(self._labeled_entry(
                    card, "Coord Z", cfg["z"],
                    partial(self._on_object_field, i, "z"),
                ))
                card_layout.addWidget(self._labeled_entry(
                    card, "Size", cfg["size"],
                    partial(self._on_object_field, i, "size"),
                ))
                card_layout.addWidget(self._labeled_entry(
                    card, "Yaw", cfg["yaw"],
                    partial(self._on_object_field, i, "yaw"),
                ))

                if cfg["kind"] == "dynamic":
                    axis_label = AXIS_INT_TO_LABEL.get(int(cfg["axis"]), "x")
                    card_layout.addWidget(self._option_menu(
                        card, "Axis", axis_label, ("x", "y", "z"),
                        partial(self._on_object_field, i, "axis"),
                    ))
                    card_layout.addWidget(self._labeled_entry(
                        card, "Min", cfg["min"],
                        partial(self._on_object_field, i, "min"),
                    ))
                    card_layout.addWidget(self._labeled_entry(
                        card, "Max", cfg["max"],
                        partial(self._on_object_field, i, "max"),
                    ))
                    card_layout.addWidget(self._labeled_entry(
                        card, "Speed", cfg["speed"],
                        partial(self._on_object_field, i, "speed"),
                    ))
                    card_layout.addWidget(self._labeled_entry(
                        card, "Dir", cfg["direction"],
                        partial(self._on_object_field, i, "direction"),
                    ))

            self.scroll_layout.addWidget(card)
        self.scroll_layout.addStretch(1)

    def _show_empty(self):
        empty = QLabel("No entries configured")
        empty.setStyleSheet("color: #666;")
        self.scroll_layout.addWidget(empty)
        self.scroll_layout.addStretch(1)

    # ---- Widget builders ----

    def _get_object_color(self, index: int) -> str:
        """Returns a hex color for an object based on its index."""
        return OBJECT_COLORS[index % len(OBJECT_COLORS)]

    def _labeled_entry(self, parent: QWidget, label: str, value, on_apply):
        row = QWidget(parent)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        lbl = QLabel(label)
        lbl.setFixedWidth(70)
        entry = QLineEdit(str(value))
        entry.returnPressed.connect(lambda e=entry: on_apply(e.text()))
        entry.editingFinished.connect(lambda e=entry: on_apply(e.text()))
        layout.addWidget(lbl)
        layout.addWidget(entry, 1)
        return row

    def _option_menu(self, parent: QWidget, label: str, current: str, options, on_apply):
        row = QWidget(parent)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        lbl = QLabel(label)
        lbl.setFixedWidth(70)
        combo = _NoScrollComboBox()
        combo.addItems(list(options))
        combo.setCurrentText(str(current))
        combo.currentTextChanged.connect(on_apply)
        layout.addWidget(lbl)
        layout.addWidget(combo, 1)
        return row

    def _remove_object_button(self, index: int):
        btn = QPushButton("X")
        btn.setFixedSize(11, 11)
        btn.setToolTip("Remove object")
        btn.setStyleSheet(
            "QPushButton {"
            "  background-color: #ff0000; color: white;"
            "  font-size: 5px; font-weight: 700; padding: 0px; border-radius: 6px;"
            "}"
        )
        btn.clicked.connect(lambda: self._on_remove_object(index))
        return btn

    # ---- Event handlers ----

    def _on_object_field(self, index: int, key: str, raw_value):
        self._config_manager.set_object_field(index, key, raw_value)
        if key == "kind":
            self.refresh()
        elif key == "name":
            self.refresh()
        name = self._config_manager.get_name("object_spawns", index)
        self.restart_requested.emit(f"Restarting after updating {name}...")

    def _on_spawn_field(self, mode: str, index: int, key: str, raw_value):
        self._config_manager.set_spawn_field(mode, index, key, raw_value)
        name = self._config_manager.get_name(mode, index)
        self.restart_requested.emit(f"Restarting after updating {name}...")

    def _on_swarm_field(self, swarm_idx: int, key: str, raw_value):
        self._config_manager.set_swarm_field(swarm_idx, key, raw_value)
        self.restart_requested.emit(f"Restarting after updating Swarm {swarm_idx + 1}...")

    def _on_drone_offset(self, drone_index: int, key: str, raw_value):
        self._config_manager.set_drone_offset_field(drone_index, key, raw_value)
        self.restart_requested.emit(f"Restarting after updating Drone {drone_index + 1}...")

    def _toggle_swarm_expand(self, swarm_idx: int):
        if swarm_idx in self._expanded_swarms:
            self._expanded_swarms.remove(swarm_idx)
        else:
            self._expanded_swarms.add(swarm_idx)
        self.refresh()

    def _toggle_object_expand(self, obj_idx: int):
        if obj_idx in self._expanded_objects:
            self._expanded_objects.remove(obj_idx)
        else:
            self._expanded_objects.add(obj_idx)
        self.refresh()

    def _set_object_filter(self, filter_type: str):
        """Sets the object filter and updates the UI."""
        self._object_filter = filter_type
        self._update_filter_buttons()
        self.refresh()

    def _update_filter_buttons(self):
        """Updates the filter button states based on current filter."""
        self.filter_btn_all.setChecked(self._object_filter == "all")
        self.filter_btn_static.setChecked(self._object_filter == "static")
        self.filter_btn_dynamic.setChecked(self._object_filter == "dynamic")

    def _on_search_text_changed(self, text: str):
        """Handler for when the search text is changed."""
        self._object_search = text
        self.refresh()

    def _on_remove_object(self, index: int):
        name = self._config_manager.get_name(self._sidebar_mode, index)
        answer = QMessageBox.question(
            self,
            "Confirm deletion",
            f"Remove {name}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        if not self._config_manager.remove_at(self._sidebar_mode, index):
            return
        new_count = self._config_manager.count(self._sidebar_mode)
        self.count_spinbox.blockSignals(True)
        self.count_spinbox.setValue(new_count)
        self.count_spinbox.blockSignals(False)
        self.refresh()
        self.restart_requested.emit(f"Restarting after removing {name}...")
