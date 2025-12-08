function run_csv_wavelet()
    % =========================================================================
    % 1. 使用者設定區 (User Settings)
    % =========================================================================
    
    % --- [模式切換] ---
    IS_SIMULATION = true;   % true = 模擬模式 (測試程式邏輯用)
                            % false = 批次模式 (跑真實實驗數據用)
    
    % --- [路徑設定] ---
    % 請修改為你的實際路徑
    SIM_FILE_PATH = "C:\Users\kein9\OneDrive\桌面\LAB\RelaxingBC_AC\matlab\SimTrap_VR_1_PreTest_EEG.csv";
    TARGET_FOLDER_PATH = "C:\Users\kein9\OneDrive\桌面\LAB";
    
    % --- [分析參數] ---
    target_fs = 200;    
    fmin = 0.5;         % 配合 Delta 下限，這裡稍微調低一點
    fmax = 60;          % 配合 Gamma 上限 (58)，這裡拉高一點
    fstep = 0.5;        
    
    % --- [頻段定義 (Hz)] - 依據你的要求修改 ---
    Bands.Delta = [0.5 4];
    Bands.Theta = [4 8];
    Bands.Alpha = [8 14];
    Bands.Beta  = [14 30];
    Bands.Gamma = [30 58];

    % --- [通道名稱] - 依據 PICO_CHANNELS_ORDER ---
    % 假設 CSV 資料欄位順序固定為: FC3, FCz, Pz, Oz
    ChannelNames = {'FC3', 'FCz', 'Pz', 'Oz'};

    % =========================================================================
    % 2. 檔案列表準備
    % =========================================================================
    if IS_SIMULATION
        fprintf('🔴 目前模式：模擬模式\n');
        if ~isfile(SIM_FILE_PATH), error('❌ 找不到模擬檔案！'); end
        [folderPath, name, ext] = fileparts(SIM_FILE_PATH);
        csvFolder = char(folderPath);
        fullFileName = [char(name), char(ext)];
        csvFiles = struct('name', fullFileName, 'folder', csvFolder, 'isdir', false);
    else
        fprintf('🔵 目前模式：資料夾批次處理\n');
        csvFolder = char(TARGET_FOLDER_PATH);
        csvFiles = dir(fullfile(csvFolder, '*.csv'));
    end

    outputRoot = fullfile(csvFolder, 'Output_Wavelet_Results');
    if ~exist(outputRoot, 'dir'), mkdir(outputRoot); end

    FinalTable = table(); % 準備總表

    % =========================================================================
    % 3. 主程式迴圈
    % =========================================================================
    fprintf('🔎 準備分析 %d 個檔案...\n', length(csvFiles));
    
    for i = 1:length(csvFiles)
        fileName = csvFiles(i).name;
        filePath = fullfile(csvFolder, fileName);
        [~, pureName, ~] = fileparts(fileName);
        
        fprintf('\n➡️ [%d/%d] 處理中: %s\n', i, length(csvFiles), fileName);
        
        % --- A. 讀檔與 B. 重取樣 ---
        try
            T = readtable(filePath, 'ReadVariableNames', false, 'FileType', 'text');
            rawEEG = table2array(T(:, 2:end)); % 假設第1欄是時間，第2欄開始是數據
            
            % 檢查通道數是否符合設定
            if size(rawEEG, 2) ~= length(ChannelNames)
                warning('⚠️ 檔案通道數 (%d) 與設定 (%d) 不符，將使用預設命名 (Ch1...)', size(rawEEG, 2), length(ChannelNames));
                currentChNames = arrayfun(@(x) sprintf('Ch%d', x), 1:size(rawEEG, 2), 'UniformOutput', false);
            else
                currentChNames = ChannelNames;
            end
            
            % 時間軸處理
            duration_sec = size(rawEEG, 1) / target_fs; 
            if size(T,2) > 1 && iscell(T.(1))
                try
                    t_start = datetime(T.(1){1}, 'InputFormat', 'yyyy-MM-dd-HH:mm:ss.SSSSSS');
                    t_end = datetime(T.(1){end}, 'InputFormat', 'yyyy-MM-dd-HH:mm:ss.SSSSSS');
                    duration_sec = seconds(t_end - t_start);
                catch
                end
            end
            
            t_old = linspace(0, duration_sec, size(rawEEG, 1));
            t_new = 0 : (1/target_fs) : duration_sec;
            data_resampled = zeros(length(t_new), size(rawEEG, 2));
            for ch = 1:size(rawEEG, 2)
                data_resampled(:, ch) = interp1(t_old, rawEEG(:, ch), t_new, 'pchip');
            end
        catch ME
            warning('讀檔錯誤: %s', ME.message);
            continue;
        end
        
        % --- 建立單檔特徵表 ---
        FileFeats = table(string(pureName), 'VariableNames', {'FileName'});
        
        % --- C. Wavelet 分析與特徵計算 ---
        numChannels = size(data_resampled, 2);
        
        for ch = 1:numChannels
            chName = currentChNames{ch}; % 取得當前通道名稱 (例如 FC3)
            signal = data_resampled(:, ch)';
            
            % 執行 Wavelet
            [TF, freqs] = tfa_morlet(signal, target_fs, fmin, fmax, fstep);
            
            % 存圖 (檔名加上通道名稱)
            saveNameBase = sprintf('%s_%s', pureName, chName);
            
            % (選擇性開啟) 存 .mat 檔太佔空間的話可註解掉下面這行
            % save(fullfile(outputRoot, [saveNameBase '.mat']), 'TF', 't_new', 'freqs');
            
            plot_spectrogram(t_new, freqs, TF, fullfile(outputRoot, [saveNameBase '.png']), saveNameBase);
            
            % --- 計算頻段能量佔比 ---
            PowerMap = TF .^ 2;
            AvgSpectrum = mean(PowerMap, 2); 
            
            bandNames = fieldnames(Bands);
            ch_powers = zeros(1, length(bandNames));
            
            for b = 1:length(bandNames)
                bName = bandNames{b};
                range = Bands.(bName);
                idx = freqs >= range(1) & freqs <= range(2);
                ch_powers(b) = sum(AvgSpectrum(idx));
            end
            
            % 計算佔比 (Ratio)
            total_power = sum(ch_powers);
            if total_power == 0, total_power = eps; end
            ch_ratios = ch_powers / total_power;
            
            % --- 存入 Table (格式: FC3_Alpha, Pz_Beta...) ---
            for b = 1:length(bandNames)
                colName = sprintf('%s_%s', chName, bandNames{b});
                FileFeats.(colName) = ch_ratios(b);
            end
        end
        
        % 合併至總表
        if isempty(FinalTable)
            FinalTable = FileFeats;
        else
            FinalTable = [FinalTable; FileFeats]; 
        end
        
        fprintf('   ✅ %s 分析完成。\n', pureName);
    end
    
    % =========================================================================
    % 4. 匯出 CSV
    % =========================================================================
    if ~isempty(FinalTable)
        outputCSV = fullfile(outputRoot, 'All_Features_Summary.csv');
        writetable(FinalTable, outputCSV);
        fprintf('\n📊 總表已匯出至: %s\n', outputCSV);
        
        if IS_SIMULATION
            disp('--- 數據預覽 (部分欄位) ---');
            disp(FinalTable(:, 1:min(6, width(FinalTable))));
        end
    end
    fprintf('\n🎉 全部工作結束！\n');
end

% =========================================================================
% 數學函式庫
% =========================================================================
function [TF, freqs] = tfa_morlet(td, fs, fmin, fmax, fstep)
    freqs = fmin:fstep:fmax;
    TF = zeros(length(freqs), length(td));
    for i = 1:length(freqs)
        fc = freqs(i);
        MW = MorletWavelet(fc/fs);
        cr = conv(td, MW, 'same');
        TF(i, :) = abs(cr);
    end
end

function [MW] = MorletWavelet(fc)
    F_RATIO = 7; Zalpha2 = 3.3; 
    sigma_f = fc/F_RATIO; sigma_t = 1/(2*pi*sigma_f);
    A = 1/sqrt(sigma_t*sqrt(pi));
    max_t = ceil(Zalpha2 * sigma_t); t = -max_t:max_t;
    v1 = 1/(-2*sigma_t^2); v2 = 2i*pi*fc;
    MW = A * exp(t.*(t.*v1+v2));
end

function plot_spectrogram(time, freqs, TF, savePath, titleText)
    h = figure('Visible', 'off');
    imagesc(time, freqs, TF); axis xy; colormap('jet');
    xlabel('Time (s)'); ylabel('Frequency (Hz)');
    title(titleText, 'Interpreter', 'none'); colorbar;
    clim([0, prctile(TF(:), 98)]);
    saveas(h, savePath); close(h);
end