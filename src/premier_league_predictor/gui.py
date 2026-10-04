from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyqtgraph as pg
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QObject, QRunnable, QSortFilterProxyModel, Qt, QThreadPool, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QListWidget, QMainWindow, QMessageBox, QProgressBar, QPushButton, QScrollArea, QSizePolicy,
    QSplitter, QTableView, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from .database import Database
from .domain import RefreshResult
from .service import refresh

APP_STYLE = """
QWidget { background:#08111f; color:#d8e2ef; font-family:'Segoe UI'; font-size:12px; }
QMainWindow { background:#060d18; }
QFrame#panel { background:#0c1727; border:1px solid #1b2a3e; border-radius:12px; }
QFrame#metric { background:#101d2f; border:1px solid #1d3048; border-radius:9px; }
QLabel#eyebrow { color:#94a3b8; font-size:10px; font-weight:600; }
QLabel#title { color:#f8fafc; font-size:24px; font-weight:700; }
QLabel#sectionTitle { color:#f8fafc; font-size:15px; font-weight:650; }
QLabel#metricValue { color:#c8ff55; font-size:21px; font-weight:700; }
QLabel#muted { color:#8391a6; }
QPushButton { background:#c8ff55; color:#08111f; border:0; border-radius:7px; padding:8px 14px; font-weight:700; }
QPushButton:hover { background:#d8ff82; }
QPushButton:disabled { background:#40502d; color:#87916e; }
QLineEdit,QComboBox { background:#0b1626; border:1px solid #263a55; border-radius:7px; padding:7px; selection-background-color:#7447ff; }
QTabWidget::pane { border:1px solid #1b2a3e; border-radius:8px; background:#0c1727; }
QTabBar::tab { background:#0a1422; color:#93a4ba; padding:8px 14px; border:0; }
QTabBar::tab:selected { color:#c8ff55; background:#142139; }
QTableView,QTableWidget { background:#0c1727; alternate-background-color:#0a1422; border:0; gridline-color:#17283d; selection-background-color:#243d58; selection-color:#ffffff; }
QHeaderView::section { background:#101d2f; color:#8fa0b6; border:0; border-bottom:1px solid #263a55; padding:7px; font-size:10px; font-weight:650; }
QProgressBar { background:#0a1422; border:1px solid #20344e; border-radius:5px; color:#e2e8f0; text-align:center; min-height:15px; }
QProgressBar::chunk { background:#7447ff; border-radius:4px; }
QSplitter::handle { background:#111e30; width:2px; }
QScrollBar:vertical { background:#08111f; width:10px; }
QScrollBar::handle:vertical { background:#263a55; border-radius:5px; min-height:24px; }
QToolTip { background:#101d2f; color:#f8fafc; border:1px solid #344866; }
"""


class ForecastTableModel(QAbstractTableModel):
    columns = (
        ("Player", "name"), ("Club", "club"), ("Pos", "position"), ("Opp", "opponents"),
        ("xG", "expected_goals"), ("xA", "expected_assists"), ("Def", "defensive_actions"),
        ("Def %", "defensive_return_probability"), ("CS %", "clean_sheet_probability"),
        ("Start %", "start_probability"), ("Mins", "expected_minutes"),
    )

    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        super().__init__()
        self.rows = rows or []

    def set_rows(self, rows: list[dict[str, Any]]) -> None:
        self.beginResetModel()
        self.rows = rows
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self.columns)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole) -> Any:
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.columns[section][0]
        return None

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if not index.isValid():
            return None
        row = self.rows[index.row()]
        key = self.columns[index.column()][1]
        value = row[key]
        if role == Qt.UserRole:
            return row["player_code"]
        if role == Qt.DisplayRole:
            if key == "opponents":
                return ", ".join(value)
            if key in {"expected_goals", "expected_assists"}:
                return f"{value:.2f}"
            if key == "defensive_actions":
                return f"{value:.1f}"
            if key in {"defensive_return_probability", "clean_sheet_probability", "start_probability"}:
                return f"{100 * value:.0f}%"
            if key == "expected_minutes":
                return f"{value:.0f}"
            return str(value)
        if role == Qt.TextAlignmentRole:
            return int(Qt.AlignVCenter | (Qt.AlignLeft if index.column() < 4 else Qt.AlignRight))
        if role == Qt.ForegroundRole and key in {"expected_goals", "expected_assists", "defensive_return_probability"}:
            return QColor("#c8ff55" if key != "expected_assists" else "#53d7ff")
        return None


class ForecastProxy(QSortFilterProxyModel):
    def __init__(self) -> None:
        super().__init__()
        self.query = ""
        self.club = "All clubs"
        self.position = "All positions"
        self.setSortCaseSensitivity(Qt.CaseInsensitive)

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        model: ForecastTableModel = self.sourceModel()  # type: ignore[assignment]
        row = model.rows[source_row]
        query_match = self.query in f"{row['name']} {row['club']}".lower()
        club_match = self.club == "All clubs" or row["club"] == self.club
        position_match = self.position == "All positions" or row["position"] == self.position
        return query_match and club_match and position_match


class WorkerSignals(QObject):
    progress = Signal(str)
    result = Signal(object)


class RefreshWorker(QRunnable):
    def __init__(self, database_path: Path) -> None:
        super().__init__()
        self.database_path = database_path
        self.signals = WorkerSignals()

    def run(self) -> None:
        result = refresh(database_path=self.database_path, progress=self.signals.progress.emit)
        self.signals.result.emit(result)


class MainWindow(QMainWindow):
    def __init__(self, database_path: Path | str | None = None) -> None:
        super().__init__()
        self.database = Database(database_path)
        self.thread_pool = QThreadPool.globalInstance()
        self.players: list[dict[str, Any]] = []
        self.run: dict[str, Any] | None = None
        self.setWindowTitle("Premier League Predictor")
        self.resize(1500, 900)
        self.setMinimumSize(1024, 700)
        self._build_ui()
        self.load_data()

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(18, 16, 18, 16)
        root_layout.setSpacing(12)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        eyebrow = QLabel("PRIVATE · LOCAL ANALYTICS")
        eyebrow.setObjectName("eyebrow")
        self.title_label = QLabel("Premier League Predictor")
        self.title_label.setObjectName("title")
        self.status_label = QLabel("No completed prediction run")
        self.status_label.setObjectName("muted")
        title_box.addWidget(eyebrow)
        title_box.addWidget(self.title_label)
        title_box.addWidget(self.status_label)
        header.addLayout(title_box)
        header.addStretch()
        self.refresh_button = QPushButton("Refresh now")
        self.refresh_button.setAccessibleName("Refresh predictions now")
        self.refresh_button.clicked.connect(self.start_refresh)
        header.addWidget(self.refresh_button)
        root_layout.addLayout(header)

        self.progress_label = QLabel("")
        self.progress_label.setObjectName("muted")
        self.progress_label.hide()
        root_layout.addWidget(self.progress_label)

        self.fixture_scroll = QScrollArea()
        self.fixture_scroll.setWidgetResizable(True)
        self.fixture_scroll.setFixedHeight(82)
        self.fixture_scroll.setFrameShape(QFrame.NoFrame)
        self.fixture_container = QWidget()
        self.fixture_layout = QHBoxLayout(self.fixture_container)
        self.fixture_layout.setContentsMargins(0, 0, 0, 0)
        self.fixture_layout.setSpacing(8)
        self.fixture_scroll.setWidget(self.fixture_container)
        root_layout.addWidget(self.fixture_scroll)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_detail_panel())
        splitter.setSizes([880, 600])
        root_layout.addWidget(splitter, 1)
        self.setCentralWidget(root)

    def _panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("panel")
        return panel

    def _build_left_panel(self) -> QWidget:
        panel = self._panel()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        heading = QLabel("Gameweek rankings")
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)
        self.leader_tabs = QTabWidget()
        self.leader_tables: dict[str, QTableWidget] = {}
        for label, key in (("Goals", "expected_goals"), ("Assists", "expected_assists"), ("Defensive", "defensive_return_probability")):
            table = QTableWidget(0, 3)
            table.setHorizontalHeaderLabels(["#", "Player", "Forecast"])
            table.verticalHeader().hide()
            table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
            table.setSelectionBehavior(QTableView.SelectRows)
            table.setEditTriggers(QTableView.NoEditTriggers)
            table.cellClicked.connect(lambda row, _column, metric=key: self._leader_clicked(metric, row))
            self.leader_tables[key] = table
            self.leader_tabs.addTab(table, label)
        self.leader_tabs.setFixedHeight(260)
        layout.addWidget(self.leader_tabs)

        filters = QHBoxLayout()
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search players or clubs")
        self.search_box.setAccessibleName("Search players or clubs")
        self.search_box.textChanged.connect(self._apply_filters)
        self.club_combo = QComboBox()
        self.club_combo.setAccessibleName("Filter by club")
        self.club_combo.currentTextChanged.connect(self._apply_filters)
        self.position_combo = QComboBox()
        self.position_combo.addItems(["All positions", "GK", "DEF", "MID", "FWD"])
        self.position_combo.setAccessibleName("Filter by position")
        self.position_combo.currentTextChanged.connect(self._apply_filters)
        filters.addWidget(self.search_box, 1)
        filters.addWidget(self.club_combo)
        filters.addWidget(self.position_combo)
        layout.addLayout(filters)

        self.table_model = ForecastTableModel()
        self.proxy_model = ForecastProxy()
        self.proxy_model.setSourceModel(self.table_model)
        self.player_table = QTableView()
        self.player_table.setModel(self.proxy_model)
        self.player_table.setAlternatingRowColors(True)
        self.player_table.setSelectionBehavior(QTableView.SelectRows)
        self.player_table.setSelectionMode(QTableView.SingleSelection)
        self.player_table.setSortingEnabled(True)
        self.player_table.verticalHeader().setVisible(False)
        self.player_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.player_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.player_table.selectionModel().selectionChanged.connect(self._selection_changed)
        layout.addWidget(self.player_table, 1)
        self.count_label = QLabel("0 current players")
        self.count_label.setObjectName("muted")
        layout.addWidget(self.count_label)
        return panel

    def _build_detail_panel(self) -> QWidget:
        panel = self._panel()
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        self.detail_widget = QWidget()
        self.detail_layout = QVBoxLayout(self.detail_widget)
        self.detail_layout.setContentsMargins(16, 16, 16, 16)
        self.detail_layout.setSpacing(12)
        self.player_title = QLabel("Select a current player")
        self.player_title.setObjectName("title")
        self.player_subtitle = QLabel("Forecast details will appear here")
        self.player_subtitle.setObjectName("muted")
        self.detail_layout.addWidget(self.player_title)
        self.detail_layout.addWidget(self.player_subtitle)

        metrics = QGridLayout()
        self.metric_labels: dict[str, QLabel] = {}
        for index, (label, key) in enumerate(
            (("xG", "xg"), ("xA", "xa"), ("If starts xG", "xg_start"),
             ("If starts xA", "xa_start"), ("Expected minutes", "minutes"), ("Defensive actions", "def_actions"))
        ):
            card = QFrame()
            card.setObjectName("metric")
            card_layout = QVBoxLayout(card)
            caption = QLabel(label.upper())
            caption.setObjectName("eyebrow")
            value = QLabel("—")
            value.setObjectName("metricValue")
            card_layout.addWidget(caption)
            card_layout.addWidget(value)
            metrics.addWidget(card, index // 3, index % 3)
            self.metric_labels[key] = value
        self.detail_layout.addLayout(metrics)

        probability_title = QLabel("Appearance and event probabilities")
        probability_title.setObjectName("sectionTitle")
        self.detail_layout.addWidget(probability_title)
        self.probability_bars: dict[str, QProgressBar] = {}
        for label, key in (("Start", "start"), ("Bench", "bench"), ("No appearance", "none"),
                           ("Clean sheet", "clean"), ("Yellow card", "yellow"), ("Red card", "red")):
            row = QHBoxLayout()
            caption = QLabel(label)
            caption.setMinimumWidth(100)
            bar = QProgressBar()
            bar.setRange(0, 100)
            row.addWidget(caption)
            row.addWidget(bar, 1)
            self.detail_layout.addLayout(row)
            self.probability_bars[key] = bar

        form_title = QLabel("Recent expected output")
        form_title.setObjectName("sectionTitle")
        self.detail_layout.addWidget(form_title)
        self.form_chart = pg.PlotWidget()
        self.form_chart.setBackground("#0c1727")
        self.form_chart.showGrid(x=False, y=True, alpha=0.15)
        self.form_chart.setMinimumHeight(190)
        self.form_chart.addLegend(offset=(8, 8))
        self.detail_layout.addWidget(self.form_chart)

        fixture_title = QLabel("Fixture split")
        fixture_title.setObjectName("sectionTitle")
        self.detail_layout.addWidget(fixture_title)
        self.fixture_table = QTableWidget(0, 6)
        self.fixture_table.setHorizontalHeaderLabels(["Opponent", "Venue", "Kickoff", "xG", "xA", "Def %"])
        self.fixture_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.fixture_table.setEditTriggers(QTableView.NoEditTriggers)
        self.fixture_table.verticalHeader().hide()
        self.fixture_table.setMaximumHeight(160)
        self.detail_layout.addWidget(self.fixture_table)

        drivers_title = QLabel("Model drivers")
        drivers_title.setObjectName("sectionTitle")
        self.detail_layout.addWidget(drivers_title)
        self.drivers = QListWidget()
        self.drivers.setMaximumHeight(105)
        self.detail_layout.addWidget(self.drivers)

        history_title = QLabel("Previous Premier League matches vs opponent")
        history_title.setObjectName("sectionTitle")
        self.detail_layout.addWidget(history_title)
        self.history_table = QTableWidget(0, 7)
        self.history_table.setHorizontalHeaderLabels(["Season", "Club", "Venue", "Mins", "G", "A", "xG"])
        self.history_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.history_table.setEditTriggers(QTableView.NoEditTriggers)
        self.history_table.verticalHeader().hide()
        self.history_table.setMinimumHeight(170)
        self.detail_layout.addWidget(self.history_table)
        self.detail_layout.addStretch()
        scroll.setWidget(self.detail_widget)
        outer.addWidget(scroll)
        return panel

    def load_data(self) -> None:
        self.run, self.players, fixtures = self.database.dashboard()
        self.table_model.set_rows(self.players)
        clubs = sorted({row["club"] for row in self.players})
        current = self.club_combo.currentText()
        self.club_combo.blockSignals(True)
        self.club_combo.clear()
        self.club_combo.addItems(["All clubs", *clubs])
        if current in clubs:
            self.club_combo.setCurrentText(current)
        self.club_combo.blockSignals(False)
        self._populate_leaders()
        self._populate_fixtures(fixtures)
        self.count_label.setText(f"{len(self.players)} current players")
        if self.run:
            generated = datetime.fromisoformat(self.run["generated_at"])
            if generated.tzinfo is None:
                generated = generated.replace(tzinfo=timezone.utc)
            stale = (datetime.now(timezone.utc) - generated).days >= 8
            suffix = " · STALE" if stale else ""
            self.status_label.setText(
                f"Gameweek {self.run['gameweek']} · refreshed {generated.astimezone().strftime('%b %d, %I:%M %p')}"
                f" · {self.run['player_count']} players{suffix}"
            )
        else:
            self.status_label.setText("No completed run · choose Refresh now to build the local database")
        if self.players:
            self.player_table.selectRow(0)

    def _populate_fixtures(self, fixtures: list[dict[str, Any]]) -> None:
        while self.fixture_layout.count():
            item = self.fixture_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for fixture in fixtures:
            card = QFrame()
            card.setObjectName("metric")
            card.setMinimumWidth(155)
            layout = QVBoxLayout(card)
            matchup = QLabel(f"{fixture['home_short']}  v  {fixture['away_short']}")
            matchup.setStyleSheet("font-weight:700;color:#f8fafc")
            kickoff = QLabel(self._format_kickoff(fixture.get("kickoff_at")))
            kickoff.setObjectName("muted")
            layout.addWidget(matchup)
            layout.addWidget(kickoff)
            self.fixture_layout.addWidget(card)
        self.fixture_layout.addStretch()

    def _populate_leaders(self) -> None:
        for metric, table in self.leader_tables.items():
            eligible = self.players if metric != "defensive_return_probability" else [p for p in self.players if p["position"] != "GK"]
            rows = sorted(eligible, key=lambda item: float(item[metric]), reverse=True)[:10]
            table.setRowCount(len(rows))
            for index, player in enumerate(rows):
                value = f"{100 * player[metric]:.0f}%" if metric.endswith("probability") else f"{player[metric]:.2f}"
                values = (str(index + 1), f"{player['name']}  ·  {player['club']}", value)
                for column, text in enumerate(values):
                    item = QTableWidgetItem(text)
                    item.setData(Qt.UserRole, player["player_code"])
                    if column == 2:
                        item.setForeground(QColor("#c8ff55"))
                        item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    table.setItem(index, column, item)

    def _leader_clicked(self, metric: str, row: int) -> None:
        item = self.leader_tables[metric].item(row, 1)
        if item:
            self.select_player(item.data(Qt.UserRole))

    def select_player(self, player_code: str) -> None:
        for source_row, player in enumerate(self.table_model.rows):
            if player["player_code"] == player_code:
                source_index = self.table_model.index(source_row, 0)
                proxy_index = self.proxy_model.mapFromSource(source_index)
                if proxy_index.isValid():
                    self.player_table.selectRow(proxy_index.row())
                    self.player_table.scrollTo(proxy_index)
                else:
                    self.show_player(player)
                return

    def _selection_changed(self) -> None:
        indexes = self.player_table.selectionModel().selectedRows()
        if not indexes:
            return
        source = self.proxy_model.mapToSource(indexes[0])
        self.show_player(self.table_model.rows[source.row()])

    def show_player(self, player: dict[str, Any]) -> None:
        opponent = ", ".join(player["opponents"])
        venue = "/".join(player["venues"])
        self.player_title.setText(player["name"])
        self.player_subtitle.setText(f"{player['club']} · {player['position']} · {opponent} ({venue}) · {player['confidence']} confidence")
        values = {
            "xg": f"{player['expected_goals']:.2f}", "xa": f"{player['expected_assists']:.2f}",
            "xg_start": f"{player['goals_if_start']:.2f}", "xa_start": f"{player['assists_if_start']:.2f}",
            "minutes": f"{player['expected_minutes']:.0f}", "def_actions": f"{player['defensive_actions']:.1f}",
        }
        for key, text in values.items():
            self.metric_labels[key].setText(text)
        probabilities = {
            "start": player["start_probability"], "bench": player["bench_probability"],
            "none": player["no_appearance_probability"], "clean": player["clean_sheet_probability"],
            "yellow": player["yellow_probability"], "red": player["red_probability"],
        }
        for key, value in probabilities.items():
            self.probability_bars[key].setValue(round(100 * value))

        fixtures = player["fixtures"]
        self.fixture_table.setRowCount(len(fixtures))
        for row_index, fixture in enumerate(fixtures):
            values = (
                fixture["opponent"], fixture["venue"], self._format_kickoff(fixture.get("kickoff_at")),
                f"{fixture['expected_goals']:.2f}", f"{fixture['expected_assists']:.2f}",
                f"{100 * fixture['defensive_return_probability']:.0f}%",
            )
            for column, text in enumerate(values):
                self.fixture_table.setItem(row_index, column, QTableWidgetItem(text))

        self.drivers.clear()
        self.drivers.addItems(player["drivers"])
        recent, versus = self.database.player_history(player["player_code"], player["opponents"][0] if player["opponents"] else None)
        chronological = list(reversed(recent[:8]))
        self.form_chart.clear()
        if chronological:
            x = list(range(1, len(chronological) + 1))
            self.form_chart.plot(x, [row["expected_goals"] for row in chronological], pen=pg.mkPen("#c8ff55", width=2), symbol="o", name="xG")
            self.form_chart.plot(x, [row["expected_assists"] for row in chronological], pen=pg.mkPen("#53d7ff", width=2), symbol="o", name="xA")
        self.history_table.setRowCount(len(versus))
        for row_index, item in enumerate(versus):
            values = (
                item["season"], item["club_name"], item["venue"], f"{item['minutes']:.0f}",
                f"{item['goals']:.0f}", f"{item['assists']:.0f}", f"{item['expected_goals']:.2f}",
            )
            for column, text in enumerate(values):
                self.history_table.setItem(row_index, column, QTableWidgetItem(text))
        if not versus:
            self.history_table.setRowCount(1)
            empty = QTableWidgetItem("No recorded Premier League meetings against this opponent")
            self.history_table.setItem(0, 0, empty)
            self.history_table.setSpan(0, 0, 1, 7)

    def _apply_filters(self) -> None:
        self.proxy_model.query = self.search_box.text().strip().lower()
        self.proxy_model.club = self.club_combo.currentText() or "All clubs"
        self.proxy_model.position = self.position_combo.currentText()
        self.proxy_model.invalidateFilter()
        self.count_label.setText(f"Showing {self.proxy_model.rowCount()} of {len(self.players)} current players")

    def start_refresh(self) -> None:
        self.refresh_button.setEnabled(False)
        self.progress_label.setText("Starting refresh…")
        self.progress_label.show()
        worker = RefreshWorker(self.database.path)
        worker.signals.progress.connect(self.progress_label.setText)
        worker.signals.result.connect(self._refresh_finished)
        self.thread_pool.start(worker)

    def _refresh_finished(self, result: RefreshResult) -> None:
        self.refresh_button.setEnabled(True)
        self.progress_label.hide()
        self.load_data()
        if not result.ok:
            title = "Refresh failed — previous run preserved" if result.stale else "Refresh failed"
            QMessageBox.warning(self, title, result.message)

    @staticmethod
    def _format_kickoff(value: str | None) -> str:
        if not value:
            return "TBD"
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone().strftime("%a %I:%M %p")
        except ValueError:
            return value


def run_gui(database_path: Path | str | None = None) -> int:
    pg.setConfigOptions(antialias=True)
    application = QApplication.instance() or QApplication([])
    application.setApplicationName("Premier League Predictor")
    application.setOrganizationName("PremierLeaguePredictor")
    application.setStyle("Fusion")
    application.setStyleSheet(APP_STYLE)
    font = QFont("Segoe UI", 10)
    application.setFont(font)
    window = MainWindow(database_path)
    window.show()
    return application.exec()
