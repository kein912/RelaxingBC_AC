import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from train import Train
from sklearn.model_selection import train_test_split


class GUI:
    # self.master 實例變數，它代表了主視窗（master window）的 Tkinter Tk 物件，是整個 GUI 應用程式的根（root）
    def __init__(self, master):
        self.master = master

        self.learning_rate = tk.DoubleVar()
        self.stop_epochs = tk.IntVar()
        self.stop_accuracy = tk.DoubleVar()

        self.learning_rate.set(0.01)
        self.stop_epochs.set(50)
        self.stop_accuracy.set(0.95)

        self.train_acc_text = tk.StringVar()
        self.train_acc_text.set("")
        self.test_acc_text = tk.StringVar()
        self.test_acc_text.set("")
        self.train_dataset_text = tk.StringVar()
        self.train_dataset_text.set("")

        self.create_widgets()

    def create_widgets(self):
        self.master.title("Perceptron Training")

        # learning Rate
        ttk.Label(self.master, text="Learning Rate:").pack(side=tk.TOP, padx=5, pady=5)
        ttk.Entry(self.master, textvariable=self.learning_rate).pack(side=tk.TOP, padx=5, pady=5)

        # epochs
        ttk.Label(self.master, text="Epochs:").pack(side=tk.TOP, padx=5, pady=5)
        ttk.Entry(self.master, textvariable=self.stop_epochs).pack(side=tk.TOP, padx=5, pady=5)

        # accuracy
        ttk.Label(self.master, text="Accuracy:").pack(side=tk.TOP, padx=5, pady=5)
        ttk.Entry(self.master, textvariable=self.stop_accuracy).pack(side=tk.TOP, padx=5, pady=5)

        ttk.Label(self.master, text="若 epochs 的數量達到，或者 accuracy 達成，訓練就會停止").pack(side=tk.TOP, padx=5, pady=5)

        # dataset
        ttk.Button(self.master, text="choose dataset", command=self.choose_file).pack(pady=10)
        ttk.Label(self.master, textvariable=self.train_dataset_text).pack(side=tk.TOP, padx=5, pady=5)

        # start training button
        ttk.Button(self.master, text="Start Training", command=self.go_train_produce_result).pack(side=tk.TOP, pady=10)

        # show accuracy
        ttk.Label(self.master, textvariable=self.train_acc_text).pack(side=tk.TOP, padx=5, pady=5)
        ttk.Label(self.master, textvariable=self.test_acc_text).pack(side=tk.TOP, padx=5, pady=5)
        
        # 創建 Matplotlib 的 Figures 和 Axes
        self.fig, self.axes = plt.subplots(nrows=1, ncols=3, gridspec_kw={'width_ratios': [2, 2, 2]})

        # 創建 tk 的畫布，放入 Matplotlib 的圖形
        self.canvas = tk.Canvas(self.master)
        self.canvas.pack(side=tk.TOP, padx=2, pady=2, fill=tk.BOTH)

        # 將 Figures 轉換為 Tkinter Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH)

    def choose_file(self):
        # 打開檔案選擇對話框
        self.dataset_filepath = filedialog.askopenfilename(title="choose the dataset")
        self.train_dataset_text.set("filepath of the dataset: " + self.dataset_filepath)

    def go_train_produce_result(self):
        dataset = self.load_data(self.dataset_filepath)
        print("dataset")
        print(dataset)

        # spit dataset
        x_train, x_test, y_train, y_test = self.split_data(dataset)

        # call train.py to train model
        my_train = Train(learning_rate=self.learning_rate, stop_epochs=self.stop_epochs, stop_accuracy=self.stop_accuracy)
        my_train.train(dataset, x_train, y_train)

        # 清除先前的圖形
        self.axes[0].clear()
        self.axes[1].clear()
        self.axes[2].clear()
        test_acc = 0
        train_acc = 0

        # 用訓練完的 model 預測 training set, test set 的 accuracy
        train_predict = my_train.predict(x_train)
        train_acc = self.cal_acc(train_predict, y_train)
        self.train_acc_text.set("training accuracy: " + str(train_acc))
        
        test_predict = my_train.predict(x_test)
        test_acc = self.cal_acc(test_predict, y_test)
        self.test_acc_text.set("test accuracy: " + str(test_acc))

        # 用訓練完的 model 畫圖 training set, test set
        my_train.plot_result(x_train, y_train, x_test, y_test, self.fig, self.axes)

        self.canvas.draw()
        # self.canvas2.draw()

    def cal_acc(self, predict, target):
        # true == predict 則為 1，predict 是 一對一對的(target, precict)
        correct_predictions = sum(1 for true, predict in zip(target, predict) if true == predict)
        return correct_predictions / len(target)

    def load_data(self, file_path):
        data = np.loadtxt(file_path, delimiter=' ')
        return data

    def split_data(self, dataset):
        x = dataset[:, :-1]
        y = dataset[:, -1]
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.33, random_state=42)
        return x_train, x_test, y_train, y_test