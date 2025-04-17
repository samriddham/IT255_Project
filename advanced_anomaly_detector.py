import numpy as np
import psutil
import pandas as pd
from collections import deque
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

class AdvancedAnomalyDetector:
    def __init__(self, history_size=100):
        self.history_size = history_size
        self.process_history = deque(maxlen=history_size)
        self.scaler = StandardScaler()
        self.autoencoder = None
        self.is_trained = False
        self.reconstruction_threshold = None

        # Initialize logging
        logging.basicConfig(
            filename=f'process_monitor_{datetime.now().strftime("%Y%m%d")}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def collect_process_metrics(self):
        """Collect detailed metrics for all running processes"""
        process_metrics = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'num_threads']):
            try:
                pinfo = proc.info
                try:
                    num_connections = len(proc.connections(kind='inet'))
                except (psutil.AccessDenied, psutil.ZombieProcess):
                    num_connections = 0

                try:
                    num_files = len(proc.open_files())
                except (psutil.AccessDenied, psutil.ZombieProcess):
                    num_files = 0

                metrics = {
                    'pid': pinfo['pid'],
                    'name': pinfo['name'],
                    'cpu_percent': pinfo.get('cpu_percent', 0) or 0,
                    'memory_percent': pinfo.get('memory_percent', 0) or 0,
                    'num_threads': pinfo.get('num_threads', 0) or 0,
                    'num_connections': num_connections,
                    'num_files': num_files
                }

                process_metrics.append(metrics)

            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
                logging.warning(f"Error collecting metrics for process: {e}")
                continue

        return process_metrics

    def update_history(self):
        current_metrics = self.collect_process_metrics()
        self.process_history.append(current_metrics)
        logging.info(f"Updated process history. Current size: {len(self.process_history)}")

    def prepare_training_data(self):
        if not self.process_history:
            return None

        all_data = []
        for metrics in self.process_history:
            for proc in metrics:
                all_data.append([
                    proc['cpu_percent'],
                    proc['memory_percent'],
                    proc['num_threads'],
                    proc['num_connections'],
                    proc['num_files']
                ])

        return np.array(all_data)

    def build_autoencoder(self, input_dim):
        inp = Input(shape=(input_dim,))
        x = Dense(16, activation='relu')(inp)
        x = Dense(8, activation='relu')(x)
        x = Dense(16, activation='relu')(x)
        out = Dense(input_dim, activation='linear')(x)
        model = Model(inputs=inp, outputs=out)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def train_model(self):
        data = self.prepare_training_data()
        if data is None or len(data) < self.history_size:
            logging.warning("Insufficient data for training")
            return False

        try:
            scaled_data = self.scaler.fit_transform(data)
            input_dim = scaled_data.shape[1]
            self.autoencoder = self.build_autoencoder(input_dim)
            self.autoencoder.fit(scaled_data, scaled_data, epochs=50, batch_size=32, verbose=0)
            reconstructed = self.autoencoder.predict(scaled_data)
            reconstruction_error = np.mean(np.square(scaled_data - reconstructed), axis=1)
            self.reconstruction_threshold = np.percentile(reconstruction_error, 95)
            self.is_trained = True
            logging.info("Successfully trained autoencoder-based anomaly detection model")
            return True
        except Exception as e:
            logging.error(f"Error training autoencoder: {e}")
            return False

    def detect_anomalies(self, processes):
        if not self.is_trained or self.autoencoder is None:
            logging.warning("Autoencoder model not trained")
            return []

        try:
            current_data = []
            for proc in processes:
                current_data.append([
                    proc['cpu_percent'],
                    proc['memory_percent'],
                    proc['num_threads'],
                    proc['num_connections'],
                    proc['num_files']
                ])

            scaled_data = self.scaler.transform(current_data)
            reconstructed = self.autoencoder.predict(scaled_data)
            reconstruction_error = np.mean(np.square(scaled_data - reconstructed), axis=1)

            anomalies = []
            for i, err in enumerate(reconstruction_error):
                if err > self.reconstruction_threshold:
                    processes[i]['anomaly_reason'] = self.get_anomaly_reason(processes[i])
                    anomalies.append(processes[i])

            logging.info(f"Detected {len(anomalies)} anomalous processes")
            return anomalies

        except Exception as e:
            logging.error(f"Error during anomaly detection: {e}")
            return []

    def get_anomaly_reason(self, process):
        reasons = []
        if process['cpu_percent'] > 80:
            reasons.append("High CPU usage")
        if process['memory_percent'] > 80:
            reasons.append("High memory usage")
        if process['num_connections'] > 50:
            reasons.append("Unusual network activity")
        if process['num_threads'] > 100:
            reasons.append("High thread count")
        if process['num_files'] > 100:
            reasons.append("Many open files")
        return ", ".join(reasons) if reasons else "Unusual behavior pattern"

    def generate_report(self, anomalies):
        report = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_processes': len(self.process_history[-1]) if self.process_history else 0,
            'anomaly_count': len(anomalies),
            'anomalies': []
        }

        for proc in anomalies:
            report['anomalies'].append({
                'pid': proc['pid'],
                'name': proc['name'],
                'reason': proc['anomaly_reason'],
                'cpu_percent': proc['cpu_percent'],
                'memory_percent': proc['memory_percent'],
                'connections': proc['num_connections']
            })

        return report
