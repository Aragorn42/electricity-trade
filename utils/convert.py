import pandas as pd

def handle_excel(file_path):
    frames = []
    for sheet_index in range(3):
        df = pd.read_excel(
            file_path,
            sheet_name=sheet_index,
            engine='openpyxl',
            header=0,
            usecols=range(25)
        )
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])

        melted = df.melt(id_vars=[date_col], var_name='Hour_Raw', value_name='Price')

        def parse_hour(val):
            if isinstance(val, (int, float)):
                return int(val)
            import re
            nums = re.findall(r'\d+', str(val))
            return int(nums[0]) if nums else 0

        melted['Hour_Offset'] = melted['Hour_Raw'].apply(parse_hour)
        melted['Datetime'] = melted[date_col] + pd.to_timedelta(melted['Hour_Offset'], unit='h')

        final_df = (
            melted[['Datetime', 'Price']]
            .sort_values('Datetime')
            .reset_index(drop=True)
            .rename(columns={'Price': f'Price_{sheet_index}'})
        )
        frames.append(final_df)

    merged = frames[0].merge(frames[1], on='Datetime', how='outer').merge(frames[2], on='Datetime', how='outer')
    merged = merged.sort_values('Datetime').reset_index(drop=True)

    price_cols = [c for c in merged.columns if c.startswith('Price_')]
    merged[price_cols] = merged[price_cols].round(3)

    merged = merged.rename(columns={
        'Datetime': '时间戳',
        'Price_0': '日前电价',
        'Price_1': '实时电价',
        'Price_2': '日前实时差价'
    })

    merged.to_csv("output.csv", index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    file_path = "/home/liym/code/ElectricityTrade/electricity-trade/dataset/日前实时套利计算.xlsm"
    handle_excel(file_path)