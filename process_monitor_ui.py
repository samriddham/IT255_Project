import sys
import time
import psutil
import platform
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QTableWidget, QTableWidgetItem, QPushButton, QLabel,
                            QHeaderView, QHBoxLayout, QMessageBox)
from PyQt5.QtCore import QTimer, Qt, QRunnable, QThreadPool, pyqtSlot, QObject, pyqtSignal
from PyQt5.QtGui import QColor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from advanced_anomaly_detector import AdvancedAnomalyDetector
import json

class WorkerSignals(QObject):
    finished = pyqtSignal(dict)

class StatsWorker(QRunnable):
    def __init__(self, callback):
        super().__init__()
        self.signals = WorkerSignals()
        self.signals.finished.connect(callback)

    @pyqtSlot()
    def run(self):
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            processes = []

            for proc in psutil.process_iter(['pid', 'name', 'username', 'cpu_percent', 'status', 'create_time', 'cmdline']):
                try:
                    pinfo = proc.info
                    memory_mb = proc.memory_info().rss / 1024 / 1024
                    created = datetime.fromtimestamp(pinfo['create_time']).strftime('%Y-%m-%d %H:%M:%S')

                    processes.append({
                        'pid': pinfo['pid'],
                        'name': pinfo['name'],
                        'username': pinfo.get('username', 'N/A'),
                        'cpu': pinfo['cpu_percent'],
                        'memory': memory_mb,
                        'status': pinfo['status'],
                        'created': created,
                        'cmdline': pinfo.get('cmdline', [])
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue

            self.signals.finished.emit({
                'cpu_percent': cpu_percent,
                'memory': memory,
                'processes': processes
            })

        except Exception as e:
            print("Error in background worker:", e)

class ProcessMonitorUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real Time Anomaly Detector")
        self.setGeometry(100, 100, 1200, 800)

        self.anomaly_detector = AdvancedAnomalyDetector()
        self.suspicious_patterns = [
            "nmap", "netcat", "hydra", "tcpdump", "aircrack-ng",
            "wireshark", "john", "hashcat", "strace", "lsof",
            "gdb", "radare2", "pkexec", "iotop"
        ]

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        header_layout = QHBoxLayout()
        self.system_info_label = QLabel()
        self.system_info_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        header_layout.addWidget(self.system_info_label)
        layout.addLayout(header_layout)

        self.process_table = QTableWidget()
        self.process_table.setColumnCount(7)
        self.process_table.setHorizontalHeaderLabels([
            "ID", "Sensor", "Station", "Sensor_01", "Sensor_02", "Status", "Date Online"
        ])
        self.process_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.process_table.setAlternatingRowColors(True)
        self.process_table.setSortingEnabled(True)
        self.process_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #d3d3d3;
                background-color: white;
                alternate-background-color: #f6f6f6;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                padding: 4px;
                border: 1px solid #d3d3d3;
                font-weight: bold;
            }
        """)

        plt.style.use('seaborn-v0_8')
        self.figure, (self.cpu_ax, self.mem_ax) = plt.subplots(2, 1, figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)

        control_layout = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setObjectName("refresh_button")
        self.refresh_button.clicked.connect(self.update_data)
        control_layout.addWidget(self.refresh_button)

        self.auto_refresh_button = QPushButton("Auto Refresh")
        self.auto_refresh_button.setObjectName("auto_refresh_button")
        self.auto_refresh_button.setCheckable(True)
        self.auto_refresh_button.toggled.connect(self.toggle_auto_refresh)
        control_layout.addWidget(self.auto_refresh_button)

        self.detect_anomalies_button = QPushButton("Detect Anomalies")
        self.detect_anomalies_button.setObjectName("detect_anomalies_button")
        self.detect_anomalies_button.clicked.connect(self.check_anomalies)
        control_layout.addWidget(self.detect_anomalies_button)

        self.save_snapshot_button = QPushButton("Save Snapshot")
        self.save_snapshot_button.setObjectName("save_snapshot_button")
        self.save_snapshot_button.clicked.connect(self.save_resource_snapshot)
        control_layout.addWidget(self.save_snapshot_button)

        layout.addLayout(control_layout)
        layout.addWidget(QLabel("<b>Running Processes</b>"))
        layout.addWidget(self.process_table)
        layout.addWidget(QLabel("<b>System Resource Usage</b>"))
        layout.addWidget(self.canvas)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.status_label)

        self.cpu_history = []
        self.mem_history = []
        self.time_points = []
        self.start_time = time.time()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_data)
        self.threadpool = QThreadPool()

        self.update_data()

    def is_suspicious(self, name, cmdline):
        name = name.lower()
        cmdline = ' '.join(cmdline).lower() if cmdline else ''
        return any(pattern in name or pattern in cmdline for pattern in self.suspicious_patterns)

    def toggle_auto_refresh(self, checked):
        if checked:
            self.timer.start(3000)
            self.refresh_button.setEnabled(False)
        else:
            self.timer.stop()
            self.refresh_button.setEnabled(True)

    def update_data(self):
        worker = StatsWorker(self.on_data_ready)
        self.threadpool.start(worker)

    def on_data_ready(self, data):
        self.system_info_label.setText(
            f"CPU Usage: {data['cpu_percent']}% | "
            f"Memory: {data['memory'].used/1024/1024/1024:.1f}GB used of {data['memory'].total/1024/1024/1024:.1f}GB | "
            f"Memory Percent: {data['memory'].percent}%"
        )

        for proc in data['processes']:
            proc['suspicious'] = self.is_suspicious(proc['name'], proc['cmdline'])

        self.update_process_table(data['processes'])
        self.update_resource_graphs(data['cpu_percent'], data['memory'].percent)

    def update_process_table(self, processes):
        self.process_table.setSortingEnabled(False)
        self.process_table.setRowCount(len(processes))

        for row, process in enumerate(processes):
            items = [
                QTableWidgetItem(str(process['pid'])),
                QTableWidgetItem(process['name']),
                QTableWidgetItem(process['username']),
                QTableWidgetItem(f"{process['cpu']:.1f}"),
                QTableWidgetItem(f"{process['memory']:.1f}"),
                QTableWidgetItem(process['status']),
                QTableWidgetItem(process['created'])
            ]

            for item in items:
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)

            if process['suspicious']:
                for item in items:
                    item.setBackground(QColor("#8B0000"))
                    item.setForeground(QColor("#ffffff"))

            for col, item in enumerate(items):
                self.process_table.setItem(row, col, item)

        self.process_table.setSortingEnabled(True)

    def update_resource_graphs(self, cpu_percent, mem_percent):
        current_time = time.time() - self.start_time

        self.time_points.append(current_time)
        self.cpu_history.append(cpu_percent)
        self.mem_history.append(mem_percent)

        if len(self.time_points) > 50:
            self.time_points.pop(0)
            self.cpu_history.pop(0)
            self.mem_history.pop(0)

        self.cpu_ax.clear()
        self.mem_ax.clear()

        self.cpu_ax.plot(self.time_points[:-10] if len(self.time_points)>10 else self.time_points, self.cpu_history[:-10] if len(self.cpu_history)>10 else self.cpu_history, 'b-', label='Sensor_01')
        self.cpu_ax.set_ylabel('Sensor_01 (%)')
        self.cpu_ax.set_title('Sensor_01 Readout')
        self.cpu_ax.grid(True)
        self.cpu_ax.legend()

        self.mem_ax.plot(self.time_points, self.mem_history, 'r-', label='Sensor_02')
        self.mem_ax.set_xlabel('Time (s)')
        self.mem_ax.set_ylabel('Sensor_02 (%)')
        self.mem_ax.set_title('Sensor_02 Readout')
        self.mem_ax.grid(True)
        self.mem_ax.legend()

        self.figure.tight_layout()
        self.canvas.draw()

    def save_resource_snapshot(self):
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"resource_usage_snapshot_{timestamp}.png"
            self.figure.savefig(filename)
            self.status_label.setText(f"Snapshot saved as {filename}")
            self.status_label.setStyleSheet("color: #4CAF50;")
        except Exception as e:
            self.status_label.setText(f"Error saving snapshot: {str(e)}")
            self.status_label.setStyleSheet("color: #F44336;")

    def check_anomalies(self):
        try:
            self.anomaly_detector.update_history()
            if not self.anomaly_detector.is_trained:
                self.status_label.setText("Training anomaly detection model...")
                self.status_label.setStyleSheet("color: #2196F3;")
                if not self.anomaly_detector.train_model():
                    self.status_label.setText("Need more data to train model")
                    return

            current_metrics = self.anomaly_detector.collect_process_metrics()
            anomalies = self.anomaly_detector.detect_anomalies(current_metrics)
            report = self.anomaly_detector.generate_report(anomalies)

            report_file = f'anomaly_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)

            self.status_label.setText(
                f"Found {len(anomalies)} anomalous processes. Report saved to {report_file}"
            )
            self.status_label.setStyleSheet("color: #4CAF50;")

        except Exception as e:
            self.status_label.setText(f"Error detecting anomalies: {str(e)}")
            self.status_label.setStyleSheet("color: #F44336;")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    app.setStyleSheet("""
        QPushButton {
            font-size: 13px;
            padding: 6px 12px;
            border-radius: 5px;
            color: white;
        }

        QPushButton#refresh_button {
            background-color: #28a745;  /* Green */
        }

        QPushButton#auto_refresh_button {
            background-color: #007bff;  /* Blue */
        }

        QPushButton#detect_anomalies_button {
            background-color: #ffc107;  /* Yellow */
            color: black;
        }

        QPushButton#save_snapshot_button {
            background-color: #dc3545;  /* Red */
        }

        QPushButton:hover {
            opacity: 0.85;
        }
    """)

    window = ProcessMonitorUI()
    window.show()
    sys.exit(app.exec_())