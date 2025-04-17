#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include <ctime>
#include <dirent.h>
#include <sys/stat.h>
#include <algorithm>
#include <cstdlib>
#include <unistd.h>

class ProcessInfo {
public:
    int pid;
    std::string name;
    int ppid;
    double cpu_usage;
    size_t memory_usage;
    std::string status;
    std::time_t start_time;
    
    ProcessInfo(int p = 0) {
        pid = p;
        ppid = 0;
        cpu_usage = 0.0;
        memory_usage = 0;
    }
};

class ProcessMonitor {
private:
    std::map<int, ProcessInfo> processes;
    std::vector<std::string> suspicious_patterns;
    
    void initialize_patterns() {
        suspicious_patterns.push_back("nc");
        suspicious_patterns.push_back("netcat");
        suspicious_patterns.push_back("wireshark");
        suspicious_patterns.push_back("nmap");
        suspicious_patterns.push_back("john");
        suspicious_patterns.push_back("hashcat");
        suspicious_patterns.push_back("hydra");
    }
    
    bool is_suspicious_name(const std::string& name) {
        std::vector<std::string>::iterator it;
        for (it = suspicious_patterns.begin(); it != suspicious_patterns.end(); ++it) {
            if (name.find(*it) != std::string::npos) {
                return true;
            }
        }
        return false;
    }

    bool is_number(const std::string& s) {
        std::string::const_iterator it = s.begin();
        while (it != s.end() && std::isdigit(*it)) ++it;
        return !s.empty() && it == s.end();
    }

public:
    ProcessMonitor() {
        initialize_patterns();
    }

    void scan_processes() {
        processes.clear();
        DIR* proc_dir = opendir("/proc");
        if (proc_dir == NULL) {
            std::cerr << "Failed to open /proc directory" << std::endl;
            return;
        }

        struct dirent* entry;
        while ((entry = readdir(proc_dir)) != NULL) {
            std::string dir_name = entry->d_name;
            if (!is_number(dir_name)) continue;

            int pid = std::atoi(dir_name.c_str());
            ProcessInfo info(pid);
            
            // Read process name
            std::string comm_path = "/proc/" + dir_name + "/comm";
            std::ifstream comm_file(comm_path.c_str());
            if (comm_file.is_open()) {
                std::getline(comm_file, info.name);
            }
            
            // Read process status
            std::string status_path = "/proc/" + dir_name + "/status";
            std::ifstream status_file(status_path.c_str());
            std::string line;
            while (std::getline(status_file, line)) {
                if (line.find("PPid:") == 0) {
                    info.ppid = std::atoi(line.substr(6).c_str());
                }
                else if (line.find("VmRSS:") == 0) {
                    info.memory_usage = std::atol(line.substr(7).c_str());
                }
            }
            
            // Calculate CPU usage
            std::string stat_path = "/proc/" + dir_name + "/stat";
            std::ifstream stat_file(stat_path.c_str());
            std::string stat_content;
            if (std::getline(stat_file, stat_content)) {
                std::istringstream iss(stat_content);
                std::vector<std::string> stat_fields;
                std::string field;
                while (iss >> field) {
                    stat_fields.push_back(field);
                }
                
                if (stat_fields.size() > 21) {
                    unsigned long utime = std::atol(stat_fields[13].c_str());
                    unsigned long stime = std::atol(stat_fields[14].c_str());
                    info.cpu_usage = (utime + stime) / 100.0;
                }
            }
            
            processes[pid] = info;
        }
        closedir(proc_dir);
    }
    
    std::vector<ProcessInfo> detect_anomalies() {
        std::vector<ProcessInfo> suspicious_processes;
        
        std::map<int, ProcessInfo>::iterator it;
        for (it = processes.begin(); it != processes.end(); ++it) {
            bool is_suspicious = false;
            
            // Check for suspicious process names
            if (is_suspicious_name(it->second.name)) {
                is_suspicious = true;
            }
            
            // Check for high CPU usage (>80%)
            if (it->second.cpu_usage > 80.0) {
                is_suspicious = true;
            }
            
            // Check for unusual parent-child relationships
            if (it->second.ppid == 1 && is_suspicious_name(it->second.name)) {
                is_suspicious = true;
            }
            
            if (is_suspicious) {
                suspicious_processes.push_back(it->second);
            }
        }
        
        return suspicious_processes;
    }
    
    const std::map<int, ProcessInfo>& get_processes() const {
        return processes;
    }
};

// Add these functions for the Python interface
extern "C" {
    ProcessMonitor* create_monitor() {
        return new ProcessMonitor();
    }
    
    void destroy_monitor(ProcessMonitor* monitor) {
        delete monitor;
    }
    
    void scan_processes(ProcessMonitor* monitor) {
        if (monitor) {
            monitor->scan_processes();
        }
    }
}