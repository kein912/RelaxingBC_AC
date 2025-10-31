from numpy import genfromtxt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import butter, lfilter, freqz
from scipy.fft import fft, fftfreq, ifft
from scipy.interpolate import interp1d 
import math
EKG_FRE = 2048
time = 50
order = 6
fs = 30.0       
cutoff = 0.5
data_count = EKG_FRE*time


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y



b, a = butter_lowpass(cutoff, fs, order)

my_data = genfromtxt('NormalData/test10/VR_1_PostTest_EKG.csv', delimiter=',')
ekg_data = my_data[:,1]

ekg_data = ekg_data[ekg_data > -100]

ekg_test = ekg_data[:data_count]
#y = fft(ekg_test)
#y[-10000:] = 0
#y = ifft(y)
data = butter_lowpass_filter(ekg_test, cutoff, fs, order)

x = np.arange(data_count)/EKG_FRE


hrw = 0.75
fs = 2048

pddata = pd.DataFrame(data, columns = ['data'])

#pd.read_csv("jiyu/VR_1_PostTest_EKG.csv")[:data_count]

mov_avg = pddata['data'].rolling(int(hrw*fs)).mean()

avg_hr = (np.mean(pddata.data))
mov_avg = [avg_hr if math.isnan(x) else x for x in mov_avg]
mov_avg = [x+200 for x in mov_avg] 
#將平均值提高 20% 以防止繼發性心臟收縮受到干擾，在第 2 部分中，我們將動態執行此操作
pddata['hart_rollingmean'] = mov_avg

window = [] #記錄每一輪ROI範圍
peaklist = [] #紀錄最高點位置
listpos = 0

for datapoint in pddata.data:
    rollingmean = pddata.hart_rollingmean[listpos]
    if (datapoint < rollingmean) and (len(window) < 1): 
	#未檢測到R-complex activity(因為len(window) < 1，所以目前沒有要檢查的ROI)
        listpos += 1
    elif (datapoint > rollingmean): #信號在平均之上，標記為ROI
        window.append(datapoint)
        listpos += 1
    else: #當信號將要掉到平均之下且等於平均的那一刻，回頭去找ROI範圍中最高的一點
        maximum = max(window)
        beatposition = listpos - len(window) + (window.index(max(window))) 
		#標記peak的位置
        peaklist.append(beatposition) 
        window = [] #重置window
        listpos += 1

ybeat = [pddata.data[x] for x in peaklist] 



#plt.plot(x,data)



RR_list = []
cnt = 0

while (cnt < (len(peaklist)-1)):
    RR_interval = (peaklist[cnt+1] - peaklist[cnt]) #計算兩點距離
    ms_dist = ((RR_interval / fs) * 1000.0) #轉換距離為時間單位
    RR_list.append(ms_dist)
    cnt += 1

bpm = 60000 / np.mean(RR_list) 
#60000 ms (1 minute) / average R-R interval of signal
print("Average Heart Beat is: %.01f" %bpm) 

RR_diff = [] #上述之第三點的部分
RR_sqdiff = [] #上述之第四點的部分
cnt = 1 

while (cnt < (len(RR_list)-1)): 
    RR_diff.append(abs(RR_list[cnt] - RR_list[cnt+1])) #計算連續 R-R 區間之間的差
    RR_sqdiff.append(math.pow(RR_list[cnt] - RR_list[cnt+1], 2)) #計算平方差
    cnt += 1

#print(RR_diff, RR_sqdiff)
ibi = np.mean(RR_list) # Inter Beat Interval
print("IBI:", ibi)

sdnn = np.std(RR_list) #R-R 區間的標準差
print("SDNN:", sdnn)

sdsd = np.std(RR_diff) #所有後續 R-R 區間之間差異的標準差
print("SDSD:", sdsd)

rmssd = np.sqrt(np.mean(RR_sqdiff)) #平方差列表的均值的根
print("RMSSD:", rmssd)

nn20 = [x for x in RR_diff if (x>20)] 
nn50 = [x for x in RR_diff if (x>50)] #創建超過 20、50 的所有值的列表
pnn20 = float(len(nn20)) / float(len(RR_diff)) 
pnn50 = float(len(nn50)) / float(len(RR_diff))
#計算NN20、NN50區間佔所有區間的比例
print("pNN20, pNN50:", pnn20, pnn50)


RR_x = peaklist[1:] #刪除第一個位置，因為第一個間隔分配給第二個節拍
RR_y = RR_list #Y 值等於區間長度
RR_x_new = np.linspace(RR_x[0],RR_x[-1],RR_x[-1]) 
#從第二個峰值開始創建均勻間隔的時間線，其端點和長度等於最後一個峰值的位置
f = interp1d(RR_x, RR_y, kind='cubic') 
#使用 cubic spline interpolation進行插值
#plt.title("Original and Interpolated Signal")
#plt.plot(RR_x, RR_y, label="Original", color='blue')
#plt.plot(RR_x_new, f(RR_x_new), label="Interpolated", color='red')
#plt.legend()



#plt.plot(pddata.data, alpha=0.5, color='blue') #原資料
#plt.plot(mov_avg, color ='green') #移動平均線
#plt.scatter(peaklist, ybeat, color='red') #peaks

n = len(pddata.data) 
frq = np.fft.fftfreq(len(pddata.data), d=((1/fs))) #將 bin 劃分為頻率類別
frq = frq[range(int(n/2))] #獲取頻率範圍的一側

#FFT
Y = np.fft.fft(f(RR_x_new))/n #計算 FFT
Y = Y[range(int(n/2))] #回傳 FFT 的一側

#Plot
plt.title("Frequency Spectrum of Heart Rate Variability")
plt.xlim(0,0.6) 
#將 X 軸限制為感興趣的頻率（0-0.6Hz 可見性，我們對 0.04-0.5 感興趣）
plt.ylim(0, 50)
plt.plot(frq, abs(Y))
plt.xlabel("Frequencies in Hz")


lf = np.trapz(abs(Y[(frq>=0.04) & (frq<=0.15)])) 
#Sx 介於 0.04 和 0.15Hz (LF) 之間的 lice 頻譜，並使用 NumPy 的trapz函數找到該區域
print("LF:", lf)

hf = np.trapz(abs(Y[(frq>=0.16) & (frq<=0.5)])) #0.16-0.5Hz (HF)
print("HF:", hf)
#plt.plot(RR_list, alpha=0.5, color='blue')
plt.show()

