%% 1. 清除環境
clear; clc; close all;

%% 2. 定義訊號參數
fs = 200;              % 取樣率 (Hz)
duration = 120;        % 模擬 120 秒
n_points = fs * duration;

% --- 您的 "標準答案" 功率分佈 ---
amp_alpha = 10;        % Alpha (11 Hz)
amp_beta  = 2;         % Beta (20 Hz)
% amp_noise = 0.5;       % 雜訊

%% 3. 建立時間軸 (秒)
% 建立一個 0 到 119.995 秒的向量
t_seconds = (0:n_points-1)' / fs; % ' 轉成行向量

%% 4. 建立 "複合波" (您的模擬 EEG)
wave_alpha = amp_alpha * sin(2 * pi * 11 * t_seconds);
wave_beta  = amp_beta  * sin(2 * pi * 20 * t_seconds);
% noise = amp_noise * randn(n_points, 1);
simulated_eeg = wave_alpha + wave_beta;  % + noise

%% 5. (最重要) 建立 "A2" 格式的 Timestamp
% 格式: yyyy-MM-dd-HH:mm:ss.SSSSSS

% 1. 設定一個 "開始時間" (例如今天早上 10 點)
startTime = datetime('now') - hours(8); % 用 'now' 或指定一個時間

% 2. 將 t_seconds (秒) 向量轉換為 "duration" (持續時間)
timeDuration = seconds(t_seconds);

% 3. 建立 "絕對時間" 向量
%    (開始時間 + 持續時間)
timestamp_vector = startTime + timeDuration;

% 4. 將 "絕對時間" 向量轉換為 "字串"
%    這就是您要的格式
timestamp_col = datestr(timestamp_vector, 'yyyy-mm-dd-HH:MM:ss.FFF');

% 注意: MATLAB 的 'FFF' 只能到毫秒 (3位數)
% Python 的 '%f' 可以到微秒 (6位數)
% 您的 Python 程式 (pd.to_datetime) 可以處理 3 位數或 6 位數
% 但如果您需要 *精確* 6 位數，請用以下這行替換：
% timestamp_col = datestr(timestamp_vector, 'yyyy-mm-dd-HH:MM:ss.MMMMMM');

%% 6. 匯出成 CSV 檔案
ch1_col = simulated_eeg;
ch2_col = simulated_eeg; % 讓 4 個 channel 都一樣
ch3_col = simulated_eeg;
ch4_col = simulated_eeg;

% 建立一個 Table (注意：第一欄現在是字串)
T = table(string(timestamp_col), ch1_col, ch2_col, ch3_col, ch4_col);

% 匯出成 CSV
%S1_VR_1_PreTest_EEG
writetable(T, 'S1_VR_1_PreTest_EEG.csv', 'WriteVariableNames', false);
disp('已儲存 S1_VR_1_PreTest_EEG.csv (A2 格式)');