
import pandas as pd

INPUT = "seir_cleaned_rebuilt.csv"
OUTPUT = "seir_cleaned_rebuilt.csv"   # 直接覆盖原文件

df = pd.read_csv(INPUT)
print("修改前列名:", list(df.columns))
print("修改前行数:", len(df))

# 找日期列（第一个含date的列）
date_col = None
for c in df.columns:
    if "date" in c.lower() or "日期" in c:
        date_col = c
        break
if date_col is None:
    date_col = df.columns[0]

# 找病例列（第一个含case/daily/病例/新增的列）
case_col = None
for c in df.columns:
    if "case" in c.lower() or "daily" in c.lower() or "病例" in c or "新增" in c:
        case_col = c
        break
if case_col is None:
    case_col = df.columns[1]

# 只保留两列，统一命名
new_df = pd.DataFrame({
    "Date": df[date_col],
    "Adjusted_Cases": df[case_col]
})

# 日期标准化为 YYYY-MM-DD
new_df["Date"] = pd.to_datetime(new_df["Date"]).dt.strftime("%Y-%m-%d")

new_df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
print("修改后列名:", list(new_df.columns))
print("修改后行数:", len(new_df))
print(f"完成！{OUTPUT} 已更新为两列，共 {len(new_df)} 行")
