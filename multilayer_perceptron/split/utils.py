def df_to_csv(df, path):
    path.write_text(df.to_csv())
