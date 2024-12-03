import pandas as pd
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt

# 读取数据
file_path = r"C:\Users\7iCK\Desktop\工作2\新数据\天然气cox1.xlsx"
data = pd.read_excel(file_path)

# 定义事件：天然气价格波动率超过某个阈值
event_threshold = 0.01  # 可以调整阈值
data['Event'] = (data['price_change'] > event_threshold).astype(int)

# 检查事件分布
event_count = data['Event'].sum()
total_count = len(data)
print(f"Total records: {total_count}, Event records: {event_count}")

# 确保有足够的事件数据
if event_count < 5:
    raise ValueError("Not enough events to perform survival analysis. Consider adjusting the event threshold.")

# 生存时间定义为日期序列
data['Duration'] = range(len(data))

# 构建生存分析数据框
covariates = ['Crude oil close']
df = data[['Duration', 'Event'] + covariates]

# 拟合Cox比例风险模型
cph = CoxPHFitter(penalizer=1.0)  # 使用较强的正则化项
cph.fit(df, 'Duration', event_col='Event')
cph.print_summary()

# 获取生存分析的结果
data['Baseline_Survival'] = cph.baseline_survival_.reset_index(drop=True)
data['Cumulative_Hazard'] = cph.baseline_cumulative_hazard_.reset_index(drop=True)

# 保存到新文件
output_path = r'C:\Users\7iCK\Desktop\\工作2\新数据\天然气cox3.xlsx'
data.to_excel(output_path, index=False)

# 绘制基线生存函数
plt.figure(figsize=(10, 6))
cph.plot()
plt.title("Baseline Survival Function")
plt.show()

# 绘制预测的累积风险函数
plt.figure(figsize=(10, 6))
cph.plot_partial_effects_on_outcome(covariates='Crude oil close', values=[0, 1])
plt.title("Partial Effects of Trend on Survival Function")
plt.show()
