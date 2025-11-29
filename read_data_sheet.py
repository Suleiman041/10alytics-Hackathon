import pandas as pd

print("Reading from 'Data' sheet...")

try:
    # Read from the 'Data' sheet
    df = pd.read_excel('10Alytics Hackathon- Fiscal Data.xlsx', sheet_name='Data')
    
    print(f"\n✓ SUCCESS!")
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}\n")
    print("="*100)
    print("FIRST 20 ROWS:")
    print("="*100)
    print(df.head(20).to_string())
    
    print("\n" + "="*100)
    print("DATA TYPES:")
    print("="*100)
    print(df.dtypes)
    
    print("\n" + "="*100)
    print("MISSING VALUES:")
    print("="*100)
    print(df.isnull().sum())
    
    print("\n" + "="*100)
    print("UNIQUE VALUES PER COLUMN:")
    print("="*100)
    for col in df.columns:
        print(f"{col}: {df[col].nunique()} unique values")
    
    # Save summary
    with open('data_summary.txt', 'w', encoding='utf-8') as f:
        f.write(f"Sheet: Data\n")
        f.write(f"Shape: {df.shape}\n")
        f.write(f"Columns: {list(df.columns)}\n\n")
        f.write("First 20 rows:\n")
        f.write(df.head(20).to_string())
    
    print("\n✓ Summary saved to 'data_summary.txt'")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
