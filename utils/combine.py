import os
import glob
import pandas as pd

def process_data():
    # 1. 获取所有文件并按字典序排序
    search_pattern = os.path.join(DATA_FOLDER, FILE_EXTENSION)
    files = sorted(glob.glob(search_pattern))
    
    if not files:
        print("未找到任何文件，请检查文件夹路径和文件后缀。")
        return

    print(f"共找到 {len(files)} 个文件，按照字典序排序完毕。")
    
    all_extracted_data = []

    # 2. 将排序后的文件与目标月份一一对应进行遍历
    # zip 会在较短的列表耗尽时停止（比如 12个文件但只有 10个月份，只会处理前 10个文件）
    for file_path, (target_year, target_month) in zip(files, TARGET_MONTHS):
        print(f"正在处理文件: {os.path.basename(file_path)} -> 提取目标: {target_year}年{target_month}月")
        
        # 为了节省内存和加快速度，只读取需要的两列 (usecols)
        try:
            df = pd.read_excel(file_path, usecols=[TIME_COL, TARGET_COL])
        except ValueError as e:
            print(f"读取 {file_path} 失败，请检查列名是否存在: {e}")
            continue

        # 3. 将时间戳列转换为 pandas 的 datetime 对象
        df[TIME_COL] = pd.to_datetime(df[TIME_COL])
        
        # 4. 筛选出特定年月的数据
        mask = (df[TIME_COL].dt.year == target_year) & (df[TIME_COL].dt.month == target_month)
        filtered_df = df[mask]
        
        # 5. 将筛选后的结果存入列表
        if not filtered_df.empty:
            all_extracted_data.append(filtered_df)
        else:
            print(f"  警告: {os.path.basename(file_path)} 中没有 {target_year}年{target_month}月 的数据！")

    # 6. 将所有月份的数据拼接成一个长文件
    if all_extracted_data:
        final_df = pd.concat(all_extracted_data, ignore_index=True)
        
        # 按照时间戳进行全局排序（可选，按需保留）
        final_df = final_df.sort_values(by=TIME_COL).reset_index(drop=True)
        
        # 导出为新的 CSV 文件
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n处理完成！总共提取了 {len(final_df)} 条数据，已保存至 {OUTPUT_FILE}")
    else:
        print("\n未提取到任何满足条件的数据。")

def reshape_and_append():
    print(f"1. 正在读取原始文件: {INPUT_FILE} ...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"错误: 找不到 {INPUT_FILE}，请确认文件路径。")
        return

    # 确保时间戳列是 datetime 格式
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])

    # 2. (可选) 限制数据范围
    if START_DATE:
        df = df[df[TIME_COL] >= pd.to_datetime(START_DATE)]
    if END_DATE:
        df = df[df[TIME_COL] <= pd.to_datetime(END_DATE)]

    if df.empty:
        print("警告：指定时间范围内没有数据！")
        return

    print("2. 正在提取日期和小时，准备重塑数据...")
    # 提取日期 (年-月-日) 和 小时 (0-23)
    df['Date'] = df[TIME_COL].dt.date
    df['Hour'] = df[TIME_COL].dt.hour

    # 3. 使用 pivot 将数据从“长表”转为“宽表”
    # Index 是日期，Columns 是小时(0-23)，Values 是目标列
    pivot_df = df.pivot(index='Date', columns='Hour', values=TARGET_COL)

    # 4. 整理列名和格式
    # 将列名(0, 1, ..., 23) 改为更清晰的字符串，例如 "00时", "01时" 或 "Hour_00"
    # 这里我们格式化为 "00:00", "01:00" ... "23:00" 方便查阅
    pivot_df.columns = [f"{hour:02d}:00" for hour in pivot_df.columns]
    
    # 将 Date 从索引变成普通的第一列
    pivot_df.reset_index(inplace=True)
    
    # 可选：将日期格式化为好看的字符串形式 'YYYY-MM-DD' (例如 2025-04-01)
    pivot_df['Date'] = pd.to_datetime(pivot_df['Date']).dt.strftime('%Y-%m-%d')

    print("3. 数据重塑完成，准备写入外部 Excel...")
    
    # 5. 写入到另外的 Excel 文件
    # 检查目标文件是否存在
    if not os.path.exists(EXTERNAL_EXCEL):
        print(f"目标文件 {EXTERNAL_EXCEL} 不存在，将自动创建新文件。")
        pivot_df.to_excel(EXTERNAL_EXCEL, sheet_name=NEW_SHEET_NAME, index=False)
    else:
        # 如果文件存在，使用 openpyxl 引擎以追加模式 (mode='a') 写入
        # if_sheet_exists='replace' 表示如果该 Sheet 已经存在，则覆盖它；如果是 'new' 则报错或建新名
        with pd.ExcelWriter(EXTERNAL_EXCEL, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            pivot_df.to_excel(writer, sheet_name=NEW_SHEET_NAME, index=False)
            
    print(f"完成！数据已成功写入到 {EXTERNAL_EXCEL} 的 '{NEW_SHEET_NAME}' Sheet 中。")
    print(f"生成的数据预览：\n{pivot_df.head(3)}")

if __name__ == "__main__":
    # ================= 配置区域 =================
    DATA_FOLDER = './'           # 数据文件所在的文件夹路径
    FILE_EXTENSION = '20260320_*_vote.xlsx'         # 文件格式，假设为 csv
    TIME_COL = '时间戳'           # 你的时间戳列名
    TARGET_COL = '预测结果'            # 你的目标列名
    OUTPUT_FILE = 'combined.csv'     # 最终输出的长文件名

    # 定义需要提取的年月顺序（2025.4 到 2026.1 共 10 个月）
    # 如果你需要 12 个月（到2026.3），请在列表末尾自行加上 (2026, 2), (2026, 3)
    TARGET_MONTHS = [
        (2025, 4), (2025, 5), (2025, 6), 
        (2025, 7), (2025, 8), (2025, 9), 
        (2025, 10), (2025, 11), (2025, 12), 
        (2026, 1), (2026, 2)
    ]
    
    process_data()
    
    INPUT_FILE = 'combined.csv'         # 之前生成的合并长文件
    EXTERNAL_EXCEL = '/home/liym/code/ElectricityTrade/electricity-trade/dataset/2026年1-2月现货价格.xlsx'  # 要被写入的新/外部 Excel 文件
    NEW_SHEET_NAME = '预测数值'          # 写入新文件时的 Sheet 名称
    TIME_COL = '时间戳'               # 时间戳列名
    TARGET_COL = '预测结果'                # 目标数据列名
    START_DATE = None
    END_DATE = None    

    reshape_and_append()