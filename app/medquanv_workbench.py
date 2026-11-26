"""MedQuanv Workbench — Ứng dụng Desktop trực quan hóa & kiểm chứng Khóa luận (PySide6 + Matplotlib).

Giao diện trạm phân tích khoa học trung tính gồm 4 màn hình:
  1. Soi ảnh & Mô hình thấy gì (ảnh gốc / nhiễu / lọc, cổng chú ý alpha & dự đoán 7 nhánh)
  2. Phần cứng thật NVIDIA Jetson Nano B01 (số đo 10W/5W + đo sống qua SSH)
  3. Đối chứng 7 nhánh & Thống kê (kiểm định cặp, TOST ±2%, Benjamini–Hochberg FDR)
  4. Đánh giá nén 4-bit & Kiểm toán Bộ nhớ L2 (INT8 vs Interp4D vs PyTorch Conv2d)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.inference_backend import (  # noqa: E402
    MODEL_SPECS,
    WorkbenchBackend,
    load_recorded_data,
    run_live_jetson_probe,
)

BRANCH_COLORS: Dict[str, str] = {
    "ctrl": "#52525B",
    "ctrl_tanh": "#0072B2",
    "ctrl_hardclip": "#56B4E9",
    "ctrl_noise_aug": "#009E73",
    "clut_septrig": "#E69F00",
    "clut_randmlp": "#CC79A7",
    "qlut": "#D55E00",
}

BRANCH_DISPLAY: Dict[str, str] = {
    "ctrl": "CNN cổ điển (148.4K)",
    "ctrl_tanh": "CNN + tanh [-1,1]",
    "ctrl_hardclip": "CNN + cắt [-1,1]",
    "ctrl_noise_aug": "CNN + học nhiễu",
    "clut_septrig": "CLUT-SepTrig",
    "clut_randmlp": "CLUT-RandMLP",
    "qlut": "Q-LUT (Lượng tử)",
}

APP_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #F7F7F5;
    color: #18181B;
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 13px;
}
#Sidebar {
    background-color: #EFEFED;
    border-right: 1px solid #E4E4E0;
}
#BrandTitle {
    font-size: 15px;
    font-weight: 700;
    color: #18181B;
}
#BrandSub {
    font-size: 11px;
    color: #52525B;
}
QPushButton#NavBtn {
    text-align: left;
    padding: 10px 14px;
    border: none;
    border-left: 3px solid transparent;
    border-radius: 4px;
    background-color: transparent;
    color: #3F3F46;
    font-weight: 500;
}
QPushButton#NavBtn:hover {
    background-color: #E7E7E3;
    color: #18181B;
}
QPushButton#NavBtn:checked {
    background-color: #FFFFFF;
    color: #1D4ED8;
    border-left: 3px solid #1D4ED8;
    font-weight: 600;
}
QFrame#Panel {
    background-color: #FFFFFF;
    border: 1px solid #E4E4E0;
    border-radius: 6px;
}
QLabel#ViewTitle {
    font-size: 17px;
    font-weight: 700;
    color: #18181B;
}
QLabel#MutedLabel {
    color: #52525B;
    font-size: 12px;
}
QLabel#SectionHeader {
    font-size: 13px;
    font-weight: 600;
    color: #18181B;
}
QLabel#MonoValue {
    font-family: "Cascadia Mono", "Consolas", monospace;
    font-weight: 600;
    color: #18181B;
}
QComboBox, QSpinBox {
    background-color: #FFFFFF;
    border: 1px solid #D4D4CE;
    border-radius: 4px;
    padding: 4px 8px;
    min-height: 24px;
}
QComboBox:focus, QSpinBox:focus {
    border: 1px solid #1D4ED8;
}
QPushButton#ActionBtn {
    background-color: #1D4ED8;
    color: #FFFFFF;
    border: none;
    border-radius: 4px;
    padding: 6px 14px;
    font-weight: 600;
}
QPushButton#ActionBtn:hover {
    background-color: #1E40AF;
}
QPushButton#ActionBtn:disabled {
    background-color: #93C5FD;
}
QPushButton#SecondaryBtn {
    background-color: #FFFFFF;
    color: #18181B;
    border: 1px solid #D4D4CE;
    border-radius: 4px;
    padding: 6px 12px;
    font-weight: 500;
}
QPushButton#SecondaryBtn:hover {
    background-color: #F4F4F0;
}
QTableWidget {
    background-color: #FFFFFF;
    border: 1px solid #E4E4E0;
    border-radius: 4px;
    gridline-color: #EEEEEC;
    selection-background-color: #EFF6FF;
    selection-color: #18181B;
}
QHeaderView::section {
    background-color: #F4F4F1;
    color: #3F3F46;
    font-weight: 600;
    font-size: 12px;
    padding: 6px 8px;
    border: none;
    border-bottom: 1px solid #E4E4E0;
    border-right: 1px solid #EEEEEC;
}
QPlainTextEdit {
    background-color: #FAFAF8;
    border: 1px solid #E4E4E0;
    border-radius: 4px;
    font-family: "Cascadia Mono", "Consolas", monospace;
    font-size: 11px;
    color: #27272A;
}
"""


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, width: float = 8.0, height: float = 5.0, dpi: int = 100) -> None:
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor="#FFFFFF")
        super().__init__(self.fig)


class JetsonProbeThread(QThread):
    finished_probe = Signal(dict)
    failed_probe = Signal(str)

    def __init__(self, power_mode: int) -> None:
        super().__init__()
        self.power_mode = power_mode

    def run(self) -> None:
        try:
            res = run_live_jetson_probe(power_mode=self.power_mode)
            self.finished_probe.emit(res)
        except Exception as ex:
            self.failed_probe.emit(str(ex))


def _style_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D4D4CE")
    ax.spines["bottom"].set_color("#D4D4CE")
    ax.tick_params(colors="#52525B", labelsize=9)
    ax.grid(True, linestyle="--", alpha=0.45, color="#D4D4CE")


def _make_item(
    text: str,
    align=Qt.AlignCenter,
    bold: bool = False,
    color: Optional[str] = None,
    bg: Optional[str] = None,
    mono: bool = False,
) -> QTableWidgetItem:
    it = QTableWidgetItem(text)
    it.setTextAlignment(align | Qt.AlignVCenter)
    f = QFont("Cascadia Mono" if mono else "Segoe UI", 9)
    f.setBold(bold)
    it.setFont(f)
    if color:
        it.setForeground(QColor(color))
    if bg:
        it.setBackground(QColor(bg))
    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
    return it


class MedQuanvWorkbench(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MedQuanv Workbench — Trực quan hóa & Đánh giá Khóa luận (MedMNIST & Jetson Nano B01)")
        self.resize(1380, 860)
        self.setMinimumSize(1160, 720)

        self.backend = WorkbenchBackend()
        self.recorded = load_recorded_data()
        self._probe_thread: Optional[JetsonProbeThread] = None

        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QHBoxLayout(root)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        sidebar = self._build_sidebar()
        main_layout.addWidget(sidebar)

        self.stack = QStackedWidget()
        main_layout.addWidget(self.stack, 1)

        self.view_inspect = self._build_inspect_view()
        self.view_jetson = self._build_jetson_view()
        self.view_stats = self._build_stats_view()
        self.view_quant = self._build_quant_view()

        self.stack.addWidget(self.view_inspect)
        self.stack.addWidget(self.view_jetson)
        self.stack.addWidget(self.view_stats)
        self.stack.addWidget(self.view_quant)

        self._refresh_inspect()
        self._populate_jetson_view()
        self._refresh_stats_view()
        self._populate_quant_view()

    # ------------------------------------------------------------------
    # SIDEBAR
    # ------------------------------------------------------------------
    def _build_sidebar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("Sidebar")
        bar.setFixedWidth(252)
        lay = QVBoxLayout(bar)
        lay.setContentsMargins(14, 18, 14, 16)
        lay.setSpacing(6)

        title = QLabel("MedQuanv Workbench")
        title.setObjectName("BrandTitle")
        sub = QLabel("Kiểm chứng Mô hình & Phần cứng Nhúng")
        sub.setObjectName("BrandSub")
        lay.addWidget(title)
        lay.addWidget(sub)
        lay.addSpacing(14)

        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        nav_items = [
            (0, "1. Soi ảnh && Mô hình thấy gì"),
            (1, "2. Phần cứng Jetson Nano B01"),
            (2, "3. Đối chứng 7 nhánh && Thống kê"),
            (3, "4. Nén 4-bit && Bộ nhớ L2"),
        ]
        self.nav_buttons: List[QPushButton] = []
        for idx, text in nav_items:
            btn = QPushButton(text)
            btn.setObjectName("NavBtn")
            btn.setCheckable(True)
            if idx == 0:
                btn.setChecked(True)
            btn.clicked.connect(lambda _=False, i=idx: self.stack.setCurrentIndex(i))
            self.nav_group.addButton(btn, idx)
            self.nav_buttons.append(btn)
            lay.addWidget(btn)

        lay.addStretch(1)

        meta = QFrame()
        meta.setObjectName("Panel")
        ml = QVBoxLayout(meta)
        ml.setContentsMargins(10, 10, 10, 10)
        ml.setSpacing(3)
        ml.addWidget(QLabel("MSSV 22520916 · Khoa KTMT - UIT"))
        s1 = QLabel("Bảng tra INT8: 256 KiB\nTổng kiểm định: 345 (BH + TOST)\nBo mạch: ARM Cortex-A57")
        s1.setObjectName("MutedLabel")
        ml.addWidget(s1)
        lay.addWidget(meta)
        return bar

    # ------------------------------------------------------------------
    # VIEW 1: SOI ẢNH & MÔ HÌNH THẤY GÌ
    # ------------------------------------------------------------------
    def _build_inspect_view(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(18, 16, 18, 16)
        lay.setSpacing(12)

        hdr = QLabel("Soi từng ảnh y tế — Bản đồ đặc trưng, Cổng chú ý & Dự đoán cả 7 mô hình")
        hdr.setObjectName("ViewTitle")
        lay.addWidget(hdr)

        ctrl = QFrame()
        ctrl.setObjectName("Panel")
        ctrl_vbox = QVBoxLayout(ctrl)
        ctrl_vbox.setContentsMargins(12, 8, 12, 8)
        ctrl_vbox.setSpacing(6)

        row1 = QHBoxLayout()
        row1.setSpacing(10)
        row1.addWidget(QLabel("Tập dữ liệu:"))
        self.cb_ds = QComboBox()
        self.cb_ds.addItem("PneumoniaMNIST (624 ảnh phổi)", "pneumoniamnist")
        self.cb_ds.addItem("BreastMNIST (156 ảnh siêu âm vú)", "breastmnist")
        self.cb_ds.currentIndexChanged.connect(self._on_ds_changed)
        row1.addWidget(self.cb_ds)

        row1.addWidget(QLabel("Ảnh số:"))
        self.sp_sample = QSpinBox()
        self.sp_sample.setRange(0, 623)
        self.sp_sample.setValue(0)
        self.sp_sample.valueChanged.connect(self._refresh_inspect)
        row1.addWidget(self.sp_sample)

        btn_rand = QPushButton("Ngẫu nhiên")
        btn_rand.setObjectName("SecondaryBtn")
        btn_rand.clicked.connect(self._pick_random_sample)
        row1.addWidget(btn_rand)

        row1.addWidget(QLabel("Mức nhiễu đốm (σ):"))
        self.sl_sigma = QSlider(Qt.Horizontal)
        self.sl_sigma.setRange(0, 20)
        self.sl_sigma.setValue(10)
        self.sl_sigma.setFixedWidth(120)
        self.sl_sigma.valueChanged.connect(self._on_sigma_changed)
        row1.addWidget(self.sl_sigma)
        self.lbl_sigma = QLabel("0.10")
        self.lbl_sigma.setObjectName("MonoValue")
        self.lbl_sigma.setFixedWidth(36)
        row1.addWidget(self.lbl_sigma)

        self.chk_damping = QCheckBox("Bật lọc giảm nhiễu (Phase Damping)")
        self.chk_damping.setChecked(True)
        self.chk_damping.stateChanged.connect(self._refresh_inspect)
        row1.addWidget(self.chk_damping)
        row1.addStretch(1)
        ctrl_vbox.addLayout(row1)

        row2 = QHBoxLayout()
        row2.setSpacing(10)
        row2.addWidget(QLabel("Chế độ bảng tra:"))
        self.cb_lut = QComboBox()
        self.cb_lut.addItem("Interp4D (1,024 KiB FP32)", "interp4d")
        self.cb_lut.addItem("INT8 Shift (256 KiB)", "int8_shift")
        self.cb_lut.currentIndexChanged.connect(self._refresh_inspect)
        row2.addWidget(self.cb_lut)

        row2.addWidget(QLabel("Hàng 2 (Chính):"))
        self.cb_prim = QComboBox()
        for mk, lbl, _ in MODEL_SPECS:
            self.cb_prim.addItem(lbl, mk)
        self.cb_prim.setCurrentIndex(0)
        self.cb_prim.currentIndexChanged.connect(self._refresh_inspect)
        row2.addWidget(self.cb_prim)

        row2.addWidget(QLabel("Hàng 3 (So sánh):"))
        self.cb_cmp = QComboBox()
        for mk, lbl, _ in MODEL_SPECS:
            self.cb_cmp.addItem(lbl, mk)
        self.cb_cmp.setCurrentIndex(4)  # ctrl_tanh
        self.cb_cmp.currentIndexChanged.connect(self._refresh_inspect)
        row2.addWidget(self.cb_cmp)
        row2.addStretch(1)
        ctrl_vbox.addLayout(row2)
        lay.addWidget(ctrl)

        splitter = QSplitter(Qt.Horizontal)

        left_panel = QFrame()
        left_panel.setObjectName("Panel")
        ll = QVBoxLayout(left_panel)
        ll.setContentsMargins(10, 10, 10, 10)
        self.canvas_inspect = MplCanvas(width=7.8, height=6.0, dpi=100)
        ll.addWidget(self.canvas_inspect)
        splitter.addWidget(left_panel)

        right_panel = QFrame()
        right_panel.setObjectName("Panel")
        rl = QVBoxLayout(right_panel)
        rl.setContentsMargins(14, 14, 14, 14)
        rl.setSpacing(10)

        self.lbl_gt = QLabel("Nhãn đúng của bác sĩ: —")
        self.lbl_gt.setObjectName("SectionHeader")
        rl.addWidget(self.lbl_gt)

        self.lbl_inspect_note = QLabel("Kết quả chẩn đoán của cả 7 mô hình trên cùng ảnh đầu vào (Seed 42):")
        self.lbl_inspect_note.setObjectName("MutedLabel")
        self.lbl_inspect_note.setWordWrap(True)
        rl.addWidget(self.lbl_inspect_note)

        self.tbl_preds = QTableWidget(len(MODEL_SPECS), 5)
        self.tbl_preds.setHorizontalHeaderLabels(["Mô hình", "Dự đoán", "Xác suất (+)", "Kết quả", "Cổng α"])
        self.tbl_preds.verticalHeader().setVisible(False)
        self.tbl_preds.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for col in (1, 2, 3, 4):
            self.tbl_preds.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeToContents)
        rl.addWidget(self.tbl_preds, 1)

        note_box = QLabel(
            "Ý nghĩa: Hàng 2 và Hàng 3 đặt bản đồ đặc trưng của Q-LUT ngay cạnh mô hình cổ điển "
            "để bạn thấy rõ cả hai đều tạo ra các vùng sáng/tối tương đương nhau — lớp lượng tử "
            "không tạo ra đặc trưng 'thần kỳ' nào khác biệt."
        )
        note_box.setObjectName("MutedLabel")
        note_box.setWordWrap(True)
        rl.addWidget(note_box)

        splitter.addWidget(right_panel)
        splitter.setSizes([760, 500])
        lay.addWidget(splitter, 1)
        return w

    def _on_ds_changed(self) -> None:
        ds = self.cb_ds.currentData()
        n = self.backend.get_num_samples(ds)
        self.sp_sample.blockSignals(True)
        self.sp_sample.setRange(0, n - 1)
        self.sp_sample.setValue(0)
        self.sp_sample.blockSignals(False)
        self._refresh_inspect()

    def _pick_random_sample(self) -> None:
        ds = self.cb_ds.currentData()
        n = self.backend.get_num_samples(ds)
        self.sp_sample.setValue(int(np.random.randint(0, n)))

    def _on_sigma_changed(self) -> None:
        self.lbl_sigma.setText(f"{self.sl_sigma.value() / 100.0:.2f}")
        self._refresh_inspect()

    def _refresh_inspect(self) -> None:
        ds = self.cb_ds.currentData()
        idx = self.sp_sample.value()
        sigma = self.sl_sigma.value() / 100.0
        use_damp = self.chk_damping.isChecked()
        lut_mode = self.cb_lut.currentData()
        prim = self.cb_prim.currentData()
        cmp_k = self.cb_cmp.currentData()

        res = self.backend.inspect_sample(
            ds=ds,
            sample_idx=idx,
            sigma=sigma,
            use_damping=use_damp,
            lut_mode=lut_mode,
            primary_key=prim,
            compare_key=cmp_k,
            seed=42,
        )

        self.lbl_gt.setText(
            f"Ảnh số #{res['sample_idx']} · Nhãn đúng của bác sĩ: {res['true_name']} (y={res['true_label']})"
        )

        fig = self.canvas_inspect.fig
        fig.clear()
        gs = fig.add_gridspec(3, 4, hspace=0.34, wspace=0.16, left=0.08, right=0.98, top=0.93, bottom=0.04)

        row0_imgs = [
            (res["img_clean"], "Ảnh gốc (28x28)", "gray", None),
            (res["img_noisy"], f"Thêm nhiễu ({sigma:.2f})", "gray", None),
            (res["img_input"], "Sau lọc nhiễu" if use_damp else "Ảnh thô (không lọc)", "gray", None),
            (res["img_input"], "Vùng chú ý (α)", "gray", res["alpha_map"]),
        ]
        for c, (img, title, cmap, ov) in enumerate(row0_imgs):
            ax = fig.add_subplot(gs[0, c])
            ax.imshow(img, cmap=cmap, vmin=0, vmax=255)
            if ov is not None:
                ax.imshow(
                    np.kron(ov, np.ones((2, 2))),
                    cmap="viridis",
                    alpha=0.48,
                    vmin=0.25,
                    vmax=0.75,
                )
            ax.set_title(title, fontsize=9, fontweight="bold", color="#18181B")
            if c == 0:
                ax.set_ylabel("Đầu vào & Chú ý", fontsize=8.5, fontweight="bold", color="#3F3F46")
            ax.set_xticks([])
            ax.set_yticks([])

        pf = res["primary_fmaps"]
        vmax_p = max(1.0, float(np.max(np.abs(pf))))
        prim_name = BRANCH_DISPLAY.get(prim, prim)
        for c in range(4):
            ax = fig.add_subplot(gs[1, c])
            ax.imshow(pf[c], cmap="coolwarm", vmin=-vmax_p, vmax=vmax_p)
            ax.set_title(f"Kênh {c}", fontsize=8.5, color="#1D4ED8")
            if c == 0:
                ax.set_ylabel(prim_name, fontsize=8.5, fontweight="bold", color="#1D4ED8")
            ax.set_xticks([])
            ax.set_yticks([])

        cf = res["compare_fmaps"]
        vmax_c = max(1.0, float(np.max(np.abs(cf))))
        cmp_name = BRANCH_DISPLAY.get(cmp_k, cmp_k)
        for c in range(4):
            ax = fig.add_subplot(gs[2, c])
            ax.imshow(cf[c], cmap="coolwarm", vmin=-vmax_c, vmax=vmax_c)
            ax.set_title(f"Kênh {c}", fontsize=8.5, color="#52525B")
            if c == 0:
                ax.set_ylabel(cmp_name, fontsize=8.5, fontweight="bold", color="#52525B")
            ax.set_xticks([])
            ax.set_yticks([])

        self.canvas_inspect.draw_idle()

        for row_i, p in enumerate(res["predictions"]):
            match_txt = "Đúng" if p["correct"] else "Sai"
            match_col = "#15803D" if p["correct"] else "#B91C1C"
            match_bg = "#F0FDF4" if p["correct"] else "#FEF2F2"
            bold_row = p["key"] in (prim, cmp_k)

            self.tbl_preds.setItem(row_i, 0, _make_item(p["label"], Qt.AlignLeft, bold=bold_row))
            self.tbl_preds.setItem(row_i, 1, _make_item(p["pred_name"], Qt.AlignCenter, bold=bold_row))
            self.tbl_preds.setItem(row_i, 2, _make_item(f"{p['prob_pos']*100:.1f}%", Qt.AlignRight, mono=True))
            self.tbl_preds.setItem(row_i, 3, _make_item(match_txt, Qt.AlignCenter, bold=True, color=match_col, bg=match_bg))
            self.tbl_preds.setItem(row_i, 4, _make_item(f"{p['alpha_mean']:.3f}", Qt.AlignRight, mono=True))
        self.tbl_preds.resizeRowsToContents()

    # ------------------------------------------------------------------
    # VIEW 2: PHẦN CỨNG JETSON NANO B01
    # ------------------------------------------------------------------
    def _build_jetson_view(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(18, 16, 18, 16)
        lay.setSpacing(12)

        hdr = QLabel("Đo thực tế trên bo mạch NVIDIA Jetson Nano B01 (P3450)")
        hdr.setObjectName("ViewTitle")
        lay.addWidget(hdr)

        top_bar = QFrame()
        top_bar.setObjectName("Panel")
        top = QHBoxLayout(top_bar)
        top.setContentsMargins(12, 8, 12, 8)
        top.setSpacing(10)
        top.addWidget(QLabel("Chế độ đo SSH trực tiếp:"))
        self.cb_power = QComboBox()
        self.cb_power.addItem("10W MAXN (4 nhân @ 1479 MHz)", 0)
        self.cb_power.addItem("5W (2 nhân @ 921 MHz)", 1)
        top.addWidget(self.cb_power)

        self.btn_probe = QPushButton("Chạy đo trực tiếp trên Jetson (SSH)")
        self.btn_probe.setObjectName("ActionBtn")
        self.btn_probe.clicked.connect(self._on_run_live_probe)
        top.addWidget(self.btn_probe)

        btn_reload = QPushButton("Nạp lại số đo đã lưu")
        btn_reload.setObjectName("SecondaryBtn")
        btn_reload.clicked.connect(self._reload_recorded)
        top.addWidget(btn_reload)
        top.addStretch(1)
        lay.addWidget(top_bar)

        strip = QFrame()
        strip.setObjectName("Panel")
        sl = QHBoxLayout(strip)
        sl.setContentsMargins(16, 12, 16, 12)
        self.lbl_kpis: List[QLabel] = []
        kpi_titles = [
            "Vi xử lý thực tế",
            "RAM tối đa tiến trình (RSS)",
            "Công suất đo thật (tegrastats)",
            "Độ khớp so với PyTorch",
        ]
        for i, kt in enumerate(kpi_titles):
            col_w = QWidget()
            col_l = QVBoxLayout(col_w)
            col_l.setContentsMargins(0, 0, 0, 0)
            col_l.setSpacing(2)
            t_lbl = QLabel(kt)
            t_lbl.setObjectName("MutedLabel")
            v_lbl = QLabel("—")
            v_lbl.setObjectName("MonoValue")
            col_l.addWidget(t_lbl)
            col_l.addWidget(v_lbl)
            self.lbl_kpis.append(v_lbl)
            sl.addWidget(col_w, 1)
            if i < 3:
                sep = QFrame()
                sep.setFrameShape(QFrame.VLine)
                sep.setStyleSheet("color: #E4E4E0;")
                sl.addWidget(sep)
        lay.addWidget(strip)

        mid = QSplitter(Qt.Horizontal)

        tbl_frame = QFrame()
        tbl_frame.setObjectName("Panel")
        tl = QVBoxLayout(tbl_frame)
        tl.setContentsMargins(12, 12, 12, 12)
        tl.addWidget(QLabel("So sánh tốc độ: Máy tính x86_64 vs Jetson Nano B01 (10W & 5W) — 624 ảnh PneumoniaMNIST"))
        self.tbl_hw = QTableWidget(8, 7)
        self.tbl_hw.setHorizontalHeaderLabels(
            ["Mô hình / Chế độ", "x86_64 (FPS)", "Jetson 10W (ms)", "Jetson 10W (FPS)", "Jetson 5W (ms)", "Jetson 5W (FPS)", "Độ chính xác / Sai số"]
        )
        self.tbl_hw.verticalHeader().setVisible(False)
        self.tbl_hw.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for col in range(1, 7):
            self.tbl_hw.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeToContents)
        tl.addWidget(self.tbl_hw)
        mid.addWidget(tbl_frame)

        chart_frame = QFrame()
        chart_frame.setObjectName("Panel")
        ch_l = QVBoxLayout(chart_frame)
        ch_l.setContentsMargins(10, 10, 10, 10)
        self.canvas_hw = MplCanvas(width=5.2, height=3.8, dpi=100)
        ch_l.addWidget(self.canvas_hw)
        mid.addWidget(chart_frame)
        mid.setSizes([700, 460])
        lay.addWidget(mid, 2)

        log_frame = QFrame()
        log_frame.setObjectName("Panel")
        lg_l = QVBoxLayout(log_frame)
        lg_l.setContentsMargins(12, 10, 12, 10)
        self.lbl_hw_status = QLabel("Nhật ký đo phần cứng & công suất tegrastats:")
        self.lbl_hw_status.setObjectName("SectionHeader")
        lg_l.addWidget(self.lbl_hw_status)
        self.txt_hw_log = QPlainTextEdit()
        self.txt_hw_log.setReadOnly(True)
        self.txt_hw_log.setMaximumHeight(165)
        lg_l.addWidget(self.txt_hw_log)
        lay.addWidget(log_frame, 1)
        return w

    def _populate_jetson_view(self) -> None:
        r = self.recorded
        rss_txt = r.get("jetson_rss", "")
        hwm_m = re.search(r"peak_vmhwm_kib=(\d+)", rss_txt)
        hwm_kib = int(hwm_m.group(1)) if hwm_m else 0

        self.lbl_kpis[0].setText("ARM Cortex-A57 (4 nhân @ 1479 MHz)")
        self.lbl_kpis[1].setText(f"{hwm_kib:,} KiB ({hwm_kib/1024:.2f} MiB)" if hwm_kib else "Không rõ")
        self.lbl_kpis[2].setText("2.65 W (10W) · 2.15 W (5W)")
        self.lbl_kpis[3].setText("Khớp 100% nhãn · Sai số <= 6.20e-6")

        x86_v2 = (r.get("x86_v2") or {}).get("modes", {})
        j_v2_10 = (r.get("jetson_v2_10w") or {}).get("modes", {})
        j_v2_5 = (r.get("jetson_v2_5w") or {}).get("modes", {})

        x86_v3 = (r.get("x86_v3") or {}).get("models", {})
        j_v3_10 = (r.get("jetson_v3_10w") or {}).get("models", {})
        j_v3_5 = (r.get("jetson_v3_5w") or {}).get("models", {})

        rows = []
        for mk, lbl in [
            ("int8_shift", "v2 INT8 Shift (833.3 KiB)"),
            ("int8_grid", "v2 INT8 Grid (841.3 KiB)"),
            ("interp4d", "v2 Interp4D (1,610.3 KiB)"),
        ]:
            a, b, c = x86_v2.get(mk, {}), j_v2_10.get(mk, {}), j_v2_5.get(mk, {})
            rows.append((
                lbl,
                f"{a.get('e2e_fps', 0):.1f}",
                f"{b.get('e2e_us', 0)/1000:.2f} ms",
                f"{b.get('e2e_fps', 0):.2f}",
                f"{c.get('e2e_us', 0)/1000:.2f} ms",
                f"{c.get('e2e_fps', 0):.2f}",
                f"{b.get('rust_acc', 0):.2f}% / {b.get('max_logit_diff', 0):.2e}",
            ))
        for mk, lbl in [
            ("quanv_v3_int8", "v3 quanv_v3_int8 (406.9 KiB)"),
            ("ctrl_v3_light", "v3 ctrl_v3_light (151.8 KiB)"),
            ("ctrl_v2_full", "v2 ctrl_v2_full (577.6 KiB)"),
            ("micro_quanv", "v3 micro_quanv (263.3 KiB)"),
            ("micro_ctrl", "v3 micro_ctrl (7.7 KiB)"),
        ]:
            a, b, c = x86_v3.get(mk, {}), j_v3_10.get(mk, {}), j_v3_5.get(mk, {})
            rows.append((
                lbl,
                f"{a.get('e2e_fps', 0):,.1f}",
                f"{b.get('e2e_us', 0)/1000:.2f} ms",
                f"{b.get('e2e_fps', 0):,.2f}",
                f"{c.get('e2e_us', 0)/1000:.2f} ms",
                f"{c.get('e2e_fps', 0):,.2f}",
                f"{b.get('rust_acc_clean', 0):.2f}% / {b.get('max_logit_diff_clean', 0):.2e}",
            ))

        self.tbl_hw.setRowCount(len(rows))
        for ri, row in enumerate(rows):
            for ci, val in enumerate(row):
                self.tbl_hw.setItem(ri, ci, _make_item(val, Qt.AlignLeft if ci == 0 else Qt.AlignRight, mono=(ci > 0)))
        self.tbl_hw.resizeRowsToContents()

        fig = self.canvas_hw.fig
        fig.clear()
        ax = fig.add_subplot(111)
        _style_ax(ax)
        labels = ["v2 INT8", "v2 Interp", "v3 Q-LUT", "v3 CNN"]
        fps10 = [
            j_v2_10.get("int8_shift", {}).get("e2e_fps", 0),
            j_v2_10.get("interp4d", {}).get("e2e_fps", 0),
            j_v3_10.get("quanv_v3_int8", {}).get("e2e_fps", 0),
            j_v3_10.get("ctrl_v3_light", {}).get("e2e_fps", 0),
        ]
        fps5 = [
            j_v2_5.get("int8_shift", {}).get("e2e_fps", 0),
            j_v2_5.get("interp4d", {}).get("e2e_fps", 0),
            j_v3_5.get("quanv_v3_int8", {}).get("e2e_fps", 0),
            j_v3_5.get("ctrl_v3_light", {}).get("e2e_fps", 0),
        ]
        x = np.arange(len(labels))
        w = 0.34
        ax.bar(x - w / 2, fps10, width=w, color="#1D4ED8", label="Chế độ 10W MAXN")
        ax.bar(x + w / 2, fps5, width=w, color="#94A3B8", label="Chế độ 5W")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_ylabel("Tốc độ xử lý (ảnh/giây - FPS)")
        ax.set_title("Tốc độ thực tế trên Jetson Nano B01", fontsize=9.5, fontweight="bold")
        ax.legend(loc="upper left", frameon=False, fontsize=8.5)
        fig.subplots_adjust(left=0.16, right=0.96, top=0.88, bottom=0.14)
        self.canvas_hw.draw_idle()

        t10 = "\n".join(r.get("tegrastats_10w", "").splitlines()[:2])
        t5 = "\n".join(r.get("tegrastats_5w", "").splitlines()[:2])
        log_preview = (
            f"[tegrastats tải 10W MAXN]\n{t10}\n\n"
            f"[tegrastats tải 5W]\n{t5}\n\n"
            f"[RAM đỉnh tiến trình]\n{rss_txt.strip()}"
        )
        self.txt_hw_log.setPlainText(log_preview)

    def _reload_recorded(self) -> None:
        self.recorded = load_recorded_data()
        self._populate_jetson_view()
        self.lbl_hw_status.setText("Đã nạp lại toàn bộ số đo Jetson Nano B01 từ file.")

    def _on_run_live_probe(self) -> None:
        if self._probe_thread and self._probe_thread.isRunning():
            return
        mode = int(self.cb_power.currentData())
        self.btn_probe.setEnabled(False)
        self.lbl_hw_status.setText("Đang kết nối SSH tới 192.168.110.51 để chạy đo trực tiếp...")
        self._probe_thread = JetsonProbeThread(mode)
        self._probe_thread.finished_probe.connect(self._on_probe_done)
        self._probe_thread.failed_probe.connect(self._on_probe_fail)
        self._probe_thread.start()

    def _on_probe_done(self, res: dict) -> None:
        self.btn_probe.setEnabled(True)
        self.lbl_hw_status.setText(f"Đo trực tiếp qua SSH hoàn tất (mã thoát {res['rc']}).")
        self.txt_hw_log.setPlainText(res["stdout"] + ("\n[stderr]\n" + res["stderr"] if res["stderr"].strip() else ""))

    def _on_probe_fail(self, err: str) -> None:
        self.btn_probe.setEnabled(True)
        self.lbl_hw_status.setText(f"Không đo được qua SSH (kiểm tra dây LAN/nguồn board): {err}")

    # ------------------------------------------------------------------
    # VIEW 3: ĐỐI CHỨNG 7 NHÁNH & THỐNG KÊ
    # ------------------------------------------------------------------
    def _build_stats_view(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(18, 16, 18, 16)
        lay.setSpacing(12)

        hdr = QLabel("Đối chứng 7 mô hình & Kiểm định thống kê (TOST ±2% + Benjamini–Hochberg)")
        hdr.setObjectName("ViewTitle")
        lay.addWidget(hdr)

        ctrl = QFrame()
        ctrl.setObjectName("Panel")
        cl = QHBoxLayout(ctrl)
        cl.setContentsMargins(12, 10, 12, 10)
        cl.setSpacing(12)

        cl.addWidget(QLabel("Tập dữ liệu:"))
        self.cb_st_ds = QComboBox()
        self.cb_st_ds.addItem("PneumoniaMNIST", "pneumoniamnist")
        self.cb_st_ds.addItem("BreastMNIST", "breastmnist")
        self.cb_st_ds.currentIndexChanged.connect(self._refresh_stats_view)
        cl.addWidget(self.cb_st_ds)

        cl.addWidget(QLabel("Chế độ tiền xử lý:"))
        self.cb_st_stem = QComboBox()
        self.cb_st_stem.addItem("Có lọc nhiễu đối xứng (Stabilized)", "stabilized_stem")
        self.cb_st_stem.addItem("Ảnh thô không lọc (Raw Stem)", "raw_stem")
        self.cb_st_stem.currentIndexChanged.connect(self._refresh_stats_view)
        cl.addWidget(self.cb_st_stem)

        cl.addWidget(QLabel("Cặp so sánh thống kê:"))
        self.cb_st_pair = QComboBox()
        pairs = [
            ("qlut_vs_ctrl", "Q-LUT vs CNN cổ điển"),
            ("qlut_vs_clut_randmlp", "Q-LUT vs Bảng tra ngẫu nhiên (CLUT-RandMLP)"),
            ("qlut_vs_clut_septrig", "Q-LUT vs Bảng tra lượng giác (CLUT-SepTrig)"),
            ("qlut_vs_ctrl_tanh", "Q-LUT vs CNN + tanh [-1, 1]"),
            ("qlut_vs_ctrl_hardclip", "Q-LUT vs CNN + cắt cứng [-1, 1]"),
            ("qlut_vs_ctrl_noise_aug", "Q-LUT vs CNN học kèm nhiễu"),
            ("t42_4_vs_2", "Đính chính Bảng 4.2: (4) Lượng tử vs (2) CNN+Lọc"),
            ("t42_3_vs_1", "Đính chính Bảng 4.2: (3) Lượng tử thô vs (1) CNN thô"),
        ]
        for pk, plbl in pairs:
            self.cb_st_pair.addItem(plbl, pk)
        self.cb_st_pair.currentIndexChanged.connect(self._refresh_stats_view)
        cl.addWidget(self.cb_st_pair)
        cl.addStretch(1)
        lay.addWidget(ctrl)

        plot_frame = QFrame()
        plot_frame.setObjectName("Panel")
        pl = QVBoxLayout(plot_frame)
        pl.setContentsMargins(10, 8, 10, 8)
        self.canvas_stats = MplCanvas(width=10.5, height=3.4, dpi=100)
        pl.addWidget(self.canvas_stats)
        lay.addWidget(plot_frame, 3)

        tbl_frame = QFrame()
        tbl_frame.setObjectName("Panel")
        tl = QVBoxLayout(tbl_frame)
        tl.setContentsMargins(10, 10, 10, 10)
        self.tbl_stats = QTableWidget(5, 9)
        self.tbl_stats.setHorizontalHeaderLabels(
            ["Nhiễu σ", "Cặp so sánh (A vs B)", "TB A", "TB B", "Chênh (A−B)", "KTC 95%", "p gốc", "p hiệu chỉnh (BH)", "Kết luận TOST (±2%)"]
        )
        self.tbl_stats.verticalHeader().setVisible(False)
        self.tbl_stats.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        for col in (0, 2, 3, 4, 5, 6, 7, 8):
            self.tbl_stats.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeToContents)
        tl.addWidget(self.tbl_stats)
        lay.addWidget(tbl_frame, 2)
        return w

    def _refresh_stats_view(self) -> None:
        rs = self.recorded.get("rigorous_stats")
        if not rs:
            return
        ds = self.cb_st_ds.currentData()
        stem = self.cb_st_stem.currentData()
        pair_key = self.cb_st_pair.currentData()

        summary = rs["seven_branch_summary"][ds][stem]
        sigmas = [0.00, 0.05, 0.10, 0.15, 0.20]
        sigma_strs = [f"{s:.2f}" for s in sigmas]

        fig = self.canvas_stats.fig
        fig.clear()
        ax = fig.add_subplot(111)
        _style_ax(ax)

        order = ["ctrl", "ctrl_tanh", "ctrl_hardclip", "ctrl_noise_aug", "clut_septrig", "clut_randmlp", "qlut"]
        for mk in order:
            ys = [summary[sg][mk]["acc_mean"] for sg in sigma_strs]
            lw = 2.4 if mk == "qlut" else 1.7
            ls = "-" if mk in ("qlut", "ctrl_tanh", "ctrl_noise_aug") else "--"
            marker = "o" if mk == "qlut" else "s"
            ax.plot(sigmas, ys, marker=marker, linestyle=ls, linewidth=lw, markersize=5, color=BRANCH_COLORS[mk], label=BRANCH_DISPLAY[mk])

        stem_vi = "Có lọc nhiễu đối xứng" if stem == "stabilized_stem" else "Ảnh thô không lọc"
        ax.set_xlabel("Mức nhiễu đốm (σ)")
        ax.set_ylabel("Độ chính xác (%, trung bình 5 seeds)")
        ax.set_xticks(sigmas)
        ax.set_title(f"{ds.upper()} — {stem_vi}", fontsize=10, fontweight="bold")
        ax.legend(loc="lower left", ncol=2, frameon=False, fontsize=8.5)
        fig.tight_layout()
        self.canvas_stats.draw_idle()

        if pair_key in ("t42_4_vs_2", "t42_3_vs_1"):
            fam_id = pair_key
        else:
            fam_id = f"{ds}_interp4d_{stem}_{pair_key}"
        rows = [t for t in rs["tests"] if t["family"] == fam_id]

        self.tbl_stats.setRowCount(len(rows))
        for ri, r in enumerate(rows):
            if r["tost_equivalent"]:
                verdict = f"TƯƠNG ĐƯƠNG (p={r['p_tost']:.4f})"
                vcol, vbg = "#15803D", "#F0FDF4"
            elif r["p_raw"] < 0.05:
                d_dir = "A>B" if r["delta_mean"] > 0 else "A<B"
                verdict = f"KHÁC BIỆT ({d_dir})"
                vcol, vbg = "#B91C1C", "#FEF2F2"
            else:
                verdict = f"CHƯA ĐỦ KẾT LUẬN ({r['p_tost']:.3f})"
                vcol, vbg = "#A16207", "#FEFCE8"

            vals = [
                r["sigma"],
                r["pair"],
                f"{r['mean_a']:.2f}%",
                f"{r['mean_b']:.2f}%",
                f"{r['delta_mean']:+.2f}%",
                f"[{r['ci95_lo']:+.2f}, {r['ci95_hi']:+.2f}]",
                f"{r['p_raw']:.4f}",
                f"{r['p_adj_family']:.4f}",
                verdict,
            ]
            for ci, v in enumerate(vals):
                if ci == 8:
                    self.tbl_stats.setItem(ri, ci, _make_item(v, Qt.AlignCenter, bold=True, color=vcol, bg=vbg))
                else:
                    self.tbl_stats.setItem(ri, ci, _make_item(v, Qt.AlignLeft if ci == 1 else Qt.AlignRight, mono=(ci != 1)))
        self.tbl_stats.resizeRowsToContents()

    # ------------------------------------------------------------------
    # VIEW 4: NÉN 4-BIT & KIỂM TOÁN BỘ NHỚ L2
    # ------------------------------------------------------------------
    def _build_quant_view(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(18, 16, 18, 16)
        lay.setSpacing(12)

        hdr = QLabel("Đánh giá thiệt hại nén ảnh 4-bit (INT8 vs Interp4D) & Kiểm toán Bộ nhớ L2")
        hdr.setObjectName("ViewTitle")
        lay.addWidget(hdr)

        top_frame = QFrame()
        top_frame.setObjectName("Panel")
        tl = QVBoxLayout(top_frame)
        tl.setContentsMargins(10, 10, 10, 10)
        self.canvas_quant = MplCanvas(width=10.5, height=4.4, dpi=100)
        tl.addWidget(self.canvas_quant)
        lay.addWidget(top_frame, 3)

        bot_frame = QFrame()
        bot_frame.setObjectName("Panel")
        bl = QVBoxLayout(bot_frame)
        bl.setContentsMargins(12, 10, 12, 10)
        bl.addWidget(QLabel("Bảng phân bổ bộ nhớ chính xác theo byte (64-bit, Batch=1, ảnh 28×28)"))
        self.tbl_mem = QTableWidget(6, 6)
        self.tbl_mem.setHorizontalHeaderLabels(
            ["Cấu hình mô hình", "Bảng Q-LUT (KiB)", "Bảng chỉ mục", "Trọng số gộp BN", "Tổng tĩnh (% L2 2MB)", "Tĩnh + Đệm đỉnh (110.25 KiB)"]
        )
        self.tbl_mem.verticalHeader().setVisible(False)
        self.tbl_mem.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for col in range(1, 6):
            self.tbl_mem.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeToContents)
        bl.addWidget(self.tbl_mem)
        lay.addWidget(bot_frame, 2)
        return w

    def _populate_quant_view(self) -> None:
        rs = self.recorded.get("rigorous_stats")
        hr = self.recorded.get("highres")
        fig = self.canvas_quant.fig
        fig.clear()
        gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 0.75], hspace=0.45, wspace=0.22, left=0.07, right=0.97, top=0.92, bottom=0.09)

        ax1 = fig.add_subplot(gs[0, 0])
        _style_ax(ax1)
        if rs:
            sigmas = [0.0, 0.05, 0.10, 0.15, 0.20]
            ds_styles = [
                ("breastmnist", "BreastMNIST", "#009E73"),
                ("pneumoniamnist", "PneumoniaMNIST", "#0072B2"),
                ("octmnist", "OCTMNIST", "#D55E00"),
            ]
            ax1.axhspan(-2.0, 2.0, color="#F0FDF4", alpha=0.9, label="Vùng tương đương TOST ±2%")
            ax1.axhline(0.0, color="#71717A", linewidth=0.9)
            for ds, lbl, col in ds_styles:
                rows = [t for t in rs["tests"] if t["family"] == f"quant_shift_vs_interp4d_{ds}"]
                deltas = [r["delta_mean"] for r in rows]
                ax1.plot(sigmas, deltas, marker="o", linewidth=1.8, color=col, label=lbl)
            ax1.set_xticks(sigmas)
            ax1.set_xlabel("Mức nhiễu (σ)")
            ax1.set_ylabel("Chênh lệch ACC (INT8 − Interp4D) %")
            ax1.set_title("Mất mát độ chính xác khi nén 4-bit (không train lại)", fontsize=9.5, fontweight="bold")
            ax1.legend(loc="lower right", frameon=False, fontsize=8)

        ax2 = fig.add_subplot(gs[0, 1])
        _style_ax(ax2)
        if hr:
            res_list = [r for r in hr["resolutions"] if r["stride"] == 2]
            x_labels = [r["resolution"] for r in res_list]
            x = np.arange(len(x_labels))
            ax2.plot(x, [r["qlut_int8_ms"] for r in res_list], marker="o", color="#009E73", linewidth=1.8, label="rust_qlut INT8 (x86_64)")
            ax2.plot(x, [r["pytorch_conv2d_ms"] for r in res_list], marker="s", color="#18181B", linewidth=1.8, label="PyTorch Conv2d (x86_64)")
            ax2.plot(x, [r["qlut_interp4d_ms"] for r in res_list], marker="^", color="#D55E00", linewidth=1.8, label="rust_qlut Interp4D (x86_64)")
            ax2.set_yscale("log")
            ax2.set_xticks(x)
            ax2.set_xticklabels(x_labels, rotation=15)
            ax2.set_ylabel("Độ trễ (ms, thang log)")
            ax2.set_title("Tốc độ 1 lớp 2×2 so với PyTorch Conv2d (không so PennyLane)", fontsize=9.5, fontweight="bold")
            ax2.legend(loc="upper left", frameon=False, fontsize=8)

        ax3 = fig.add_subplot(gs[1, :])
        _style_ax(ax3)
        y_pos = np.arange(3)
        names = ["v3 quanv_int8", "v2 INT8 Shift", "v2 Interp4D"]
        lut_s = np.array([256.0, 256.0, 1024.0])
        idx_s = np.array([0.0, 0.0, 9.0])
        w_s = np.array([150.90, 577.26, 577.26])
        act_s = np.array([0.0, 110.25, 110.25])

        ax3.barh(y_pos, lut_s, color="#1D4ED8", height=0.48, label="Bảng Q-LUT")
        ax3.barh(y_pos, idx_s, left=lut_s, color="#E69F00", height=0.48, label="Bảng chỉ mục")
        ax3.barh(y_pos, w_s, left=lut_s + idx_s, color="#52525B", height=0.48, label="Trọng số gộp BN")
        ax3.barh(y_pos, act_s, left=lut_s + idx_s + w_s, color="#93C5FD", height=0.48, label="Đệm kích hoạt đỉnh (v2)")
        ax3.axvline(2048.0, color="#B91C1C", linestyle="--", linewidth=1.4, label="Giới hạn L2 (2,048 KiB)")
        ax3.set_xlim(0, 2220)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(names, fontsize=9)
        ax3.set_xlabel("Dung lượng (KiB, so với 2,048 KiB L2 Cache của Cortex-A57)")
        ax3.legend(loc="upper center", bbox_to_anchor=(0.5, 1.32), ncol=5, frameon=False, fontsize=8)
        pos3 = ax3.get_position()
        ax3.set_position([0.13, pos3.y0, 0.84, pos3.height * 0.86])

        self.canvas_quant.draw_idle()

        mem_rows = [
            ("WideGRQuanvNet v2 — INT8 Shift-LUT", "256.00", "0.00 KiB", "577.26 KiB", "833.26 KiB (40.69%)", "943.51 KiB (46.07%)"),
            ("WideGRQuanvNet v2 — INT8 Grid-LUT", "256.00", "8.00 KiB", "577.26 KiB", "841.26 KiB (41.08%)", "951.51 KiB (46.46%)"),
            ("WideGRQuanvNet v2 — Interp4D SIMD", "1,024.00", "9.00 KiB", "577.26 KiB", "1,610.26 KiB (78.63%)", "1,720.51 KiB (84.01%)"),
            ("quanv_v3_int8 (v3, INT8 Shift)", "256.00", "0.00 KiB", "150.90 KiB", "406.90 KiB (19.87%)", ">= 406.90 KiB"),
            ("ctrl_v3_light (v3, 0% LUT)", "0.00", "0.00 KiB", "151.79 KiB", "151.79 KiB (7.41%)", ">= 151.79 KiB"),
            ("micro_quanv (v3, INT8 Shift)", "256.00", "0.00 KiB", "7.29 KiB", "263.29 KiB (12.86%)", ">= 263.29 KiB"),
        ]
        self.tbl_mem.setRowCount(len(mem_rows))
        for ri, row in enumerate(mem_rows):
            for ci, v in enumerate(row):
                self.tbl_mem.setItem(ri, ci, _make_item(v, Qt.AlignLeft if ci == 0 else Qt.AlignRight, mono=(ci > 0)))
        self.tbl_mem.resizeRowsToContents()


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLESHEET)
    win = MedQuanvWorkbench()
    win.showNormal()
    win.raise_()
    win.activateWindow()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
