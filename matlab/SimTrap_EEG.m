%% 1. 建立「陷阱」標準答案
clear; clc; close all;
fs = 200; % 假設是 200Hz (如果這個檔案本身也是 1000Hz，請改 1000)
duration = 120; 
n_points = fs * duration;
t_seconds = (0:n_points-1)' / fs;

% --- 陷阱訊號：1Hz Delta + 40Hz Gamma ---
amp_delta = 10;
amp_gamma = 10;
% amp_noise = 0.1; 
wave_delta = amp_delta * sin(2 * pi * 1 * t_seconds);   % 1 Hz (會被 Bug 1 濾掉)
wave_gamma = amp_gamma * sin(2 * pi * 40 * t_seconds); % 40 Hz (會被 Bug 1 濾掉)
simulated_eeg = wave_delta + wave_gamma; %+ (amp_noise * randn(n_points, 1));

%% 2. 轉換為 "A2" Timestamp 格式
% 1. 設定一個 "開始時間"
startTime = datetime('now') - hours(8); % 用 'now' 或指定一個時間

% 2. 將 t_seconds (秒) 向量轉換為 "duration" (持續時間)
%    (t_seconds 必須在第 3 節被定義為一個 "行" 向量, e.g., t_seconds = (0:n_points-1)' / fs;)
timeDuration = seconds(t_seconds);

% 3. (關鍵) 建立 "絕對時間" 向量 (這行可能被您誤刪了)
timestamp_vector = startTime + timeDuration;

% 4. (修正) 將 "絕對時間" 向量轉換為 "字串"
%    這才是正確的 MATLAB 語法
timestamp_col = string(timestamp_vector, 'yyyy-MM-dd-HH:mm:ss.SSSSSS');
% ----------------------------------------------------

%% 3. 匯出 CSV
T = table(string(timestamp_col), simulated_eeg, simulated_eeg, simulated_eeg, simulated_eeg);
writetable(T, 'SimTrap_VR_1_PreTest_EEG.csv', 'WriteVariableNames', false);
disp('已儲存 "陷阱" 模擬檔: SimTrap_VR_1_PreTest_EEG.csv');