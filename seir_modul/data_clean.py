
import csv
from collections import defaultdict

INPUT_FILE = "raw_data.csv"
OUTPUT_FILE = "seir_cleaned.csv"

TARGET_YEAR = 2025


OFFICIAL_TOTALS = {
    "2025-05": 440662,
    "2025-06": 333229,
    "2025-07": 226567,
    "2025-08": 164625,
    "2025-09": 66915,
}

# ==================== 主程序 ====================

def read_raw_data(filepath):

    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(",")
            if len(parts) < 2:
                continue
            date_str = parts[0].strip()
            try:
                value = float(parts[1].strip())
            except ValueError:
                print(f"警告: 无法解析数值，跳过该行: {line}")
                continue
            records.append((date_str, value))
    return records


def fix_year(date_str, target_year):

    # 把分隔符统一成 "/"，再切分
    y, m, d = date_str.replace("-", "/").split("/")
    y = target_year
    m = int(m)
    d = int(d)
    return f"{y:04d}-{m:02d}-{d:02d}"


def calibrate_by_month(records, official_totals):

    groups = defaultdict(list)
    for date, value in records:
        month = date[:7]  # 取 "YYYY-MM"
        groups[month].append([date, value])

    # 对每个月单独校准
    calibrated = []
    for month in sorted(groups.keys()):
        month_data = groups[month]
        extracted_sum = sum(v for _, v in month_data)

        if month not in official_totals:
            print(f"警告: 月份 {month} 不在官方总数表中，跳过校准，保持原值")
            scale = 1.0
        else:
            target = official_totals[month]
            scale = target / extracted_sum
            print(f"{month}: 提取总和={extracted_sum:.0f}, 官方={target}, 缩放系数={scale:.6f}")

        for date, value in month_data:
            calibrated.append((date, round(value * scale)))

    return calibrated


def write_csv(filepath, records):

    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "daily_cases"])
        for date, value in records:
            writer.writerow([date, value])
    print(f"\n已输出: {filepath}（共 {len(records)} 条记录）")


def main():
    # 1. 读取原始数据
    raw = read_raw_data(INPUT_FILE)
    print(f"读取到 {len(raw)} 条原始记录")


    fixed = [(fix_year(date, TARGET_YEAR), value) for date, value in raw]


    fixed.sort(key=lambda x: x[0])
    print(f"日期范围: {fixed[0][0]} 至 {fixed[-1][0]}")


    cleaned = calibrate_by_month(fixed, OFFICIAL_TOTALS)


    write_csv(OUTPUT_FILE, cleaned)


    peak = max(cleaned, key=lambda x: x[1])
    print(f"峰值日: {peak[0]}，峰值: {peak[1]} 例")


if __name__ == "__main__":
    main()
