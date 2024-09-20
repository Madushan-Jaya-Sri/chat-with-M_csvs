import pandas as pd

def excel_to_sql_file(file_path, table_name, output_sql_file):
    # Read the Excel file into a DataFrame
    df = pd.read_excel(file_path)

    # Clean up column names by stripping leading/trailing spaces
    df.columns = df.columns.str.strip()

    # Define column-specific data types
    data_types = {
        'Country': 'VARCHAR(255)',
        'Region': 'VARCHAR(255)',
        'Archetype': 'VARCHAR(255)',
        'Sub-archetype': 'VARCHAR(255)',
        'Investment Category': 'VARCHAR(255)',
        'Deal Type': 'VARCHAR(255)',
        'Deal Value': 'DOUBLE(20,2) DEFAULT 0.00',
        'Deal Date': 'DATETIME',
        'Deal Year': 'YEAR(4)'
    }

    # Open the SQL file to write
    with open(output_sql_file, 'w') as f:
        # Write the CREATE TABLE statement
        f.write(f'CREATE TABLE `{table_name}` (\n')
        for i, col in enumerate(df.columns):
            # Use the specified data type, default to TEXT if not listed
            col_type = data_types.get(col, 'TEXT')
            comma = ',' if i < len(df.columns) - 1 else ''
            f.write(f'    `{col}` {col_type}{comma}\n')
        f.write(') ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;\n\n')

        # Write the INSERT INTO statements
        for index, row in df.iterrows():
            # Escape single quotes in the row values
            values = "', '".join(str(value).replace("'", "''") for value in row.values)
            f.write(f"INSERT INTO `{table_name}` ({', '.join([f'`{col}`' for col in df.columns])}) VALUES ('{values}');\n")

    print(f"SQL file {output_sql_file} has been successfully generated.")

# Example usage
if __name__ == '__main__':
    # Path to your Excel file
    file_path = 'PCIT Data v3.0.1.xlsx'
    
    # Name of the SQL table
    table_name = 'investments'
    
    # Path to the output .sql file
    output_sql_file = 'PCIT.sql'
    
    excel_to_sql_file(file_path, table_name, output_sql_file)
