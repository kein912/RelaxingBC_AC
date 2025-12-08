%% 1. 建立「陷阱」標準答案 (修正版：使用頻帶雜訊)
clear; clc; close all;
fs = 200; 
duration = 120; 
n_points = fs * duration;
t_seconds = (0:n_points-1)' / fs;

% --- 產生白雜訊 ---
rng(42); % 固定亂數種子，確保每次跑結果一樣
white_noise = randn(n_points, 1);

% --- 濾波產生頻帶雜訊 (Band-limited Noise) ---
% 1. 產生 Delta 雜訊 (0.5 - 4 Hz)
% 使用 bandpass 函數 (若無 Signal Processing Toolbox 可改用 butter/filter)
try
    delta_noise = bandpass(white_noise, [0.5 4], fs);
    gamma_noise = bandpass(white_noise, [30 58], fs);
catch
    % 備用方案: 若無 bandpass 函數，使用簡單的 butterworth 濾波
    [b, a] = butter(2, [0.5 4]/(fs/2));
    delta_noise = filter(b, a, white_noise);
    [b, a] = butter(2, [30 58]/(fs/2));
    gamma_noise = filter(b, a, white_noise);
end

% --- 關鍵：能量標準化 (Normalization) ---
% 強制讓兩個訊號的「能量 (Standard Deviation)」一模一樣
target_amp = 10;
delta_noise = (delta_noise / std(delta_noise)) * target_amp;
gamma_noise = (gamma_noise / std(gamma_noise)) * target_amp;

% --- 合成訊號 ---
simulated_eeg = delta_noise + gamma_noise; 

% 檢查一下是不是真的 1:1
fprintf('Delta 能量: %.2f\n', std(delta_noise));
fprintf('Gamma 能量: %.2f\n', std(gamma_noise));

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