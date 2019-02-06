import csv
import numpy as np
import matplotlib.pyplot as plt

history2017 = []
with open('2017_EURUSD.csv', newline='') as csvfile:
    historyloader = csv.reader(csvfile)
    fxh = np.array(list(historyloader))
history2017 = np.array(fxh[:, 2], dtype=float)
history2017 = np.mean((history2017[-369000:]).reshape(-1, 15), axis=1)

history2018 = []
with open('2018_EURUSD.csv', newline='') as csvfile:
    historyloader = csv.reader(csvfile)
    fxh = np.array(list(historyloader))
history2018 = np.array(fxh[:, 2], dtype=float)
history2018 = np.mean((history2018[:372000]).reshape(-1, 15), axis=1)

history = np.concatenate((history2017.astype(float), history2018.astype(float)))


def calcSma(values, smaPeriod):
    a = np.array(values)
    b = a.view()

    for i in range(1, smaPeriod):
        b = np.roll(b, 1)
        b[0] = b[1]
        a += b
    a = a / smaPeriod
    return a


sma20 = calcSma(history, 20)
print("SMA Length" + str(len(sma20)))


def ExpMovingAverage(values, window):
    a = np.array(values)

    smoothFactor = 2.0 / (1.0 + window)

    b = np.zeros(len(a))
    b[0] = a[0]

    for i in range(1,len(b)):
        b[i] = a[i] * smoothFactor + b[i-1] * (1.0-smoothFactor)
    return (b)


ema20 = ExpMovingAverage(history, 20)
print("EMA Length" + str(len(ema20)))


def TrueRange(values, window):
    out = []
    for i in range(window, len(values)):
        val = values[i - window:i]
        out.append(abs(np.min(val) - np.max(val)))
    return out


tr20 = TrueRange(history, 20)
print("TR length" + str(len(tr20)))


def AvgTrueRange(values, window):
    return ExpMovingAverage(TrueRange(values, 20), window)


atr = AvgTrueRange(history, 14)
print("ATR length" + str(len(atr)))

print("Length of forex history array: " + str(len(history)) + " (" + str(len(history) / (24 * 4)) + " Days)")

plotperiod = 4 * 24 * 4

plt.subplot(2, 1, 1)
plt.plot(history[:plotperiod], 'r')
plt.plot(sma20[:plotperiod], 'b')
plt.plot(ema20[:plotperiod], 'g')
plt.subplot(2, 1, 2)
plt.plot(tr20[:plotperiod], 'r')
plt.plot(atr[:plotperiod], 'g')
plt.show()
