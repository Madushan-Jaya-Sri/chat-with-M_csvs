import pandas as pd


def excel_to_sql_file(file_path, table_name, output_sql_file):
    # Read the Excel file into a DataFrame
    df = pd.read_excel(file_path)

    # Clean up column names by stripping leading/trailing spaces
    df.columns = df.columns.str.strip()

    # Open the SQL file to write
    with open(output_sql_file, 'w') as f:
        # Write the CREATE TABLE statement
        columns = df.columns
        f.write(f'CREATE TABLE `{table_name}` (\n')
        for i, col in enumerate(columns):
            # Write column definitions, without adding a comma to the last column
            comma = ',' if i < len(columns) - 1 else ''
            f.write(f'    `{col}` TEXT{comma}\n')
        f.write(');\n\n')

        # Write the INSERT INTO statements
        for index, row in df.iterrows():
            # Escape single quotes in the row values
            values = "', '".join(str(value).replace("'", "''") for value in row.values)
            f.write(f"INSERT INTO `{table_name}` ({', '.join([f'`{col}`' for col in columns])}) VALUES ('{values}');\n")

    print(f"SQL file {output_sql_file} has been successfully generated.")

# Example usage
if __name__ == '__main__':
    # Path to your Excel file
    file_path = 'PCIT Data v3.0.1.xlsx'
    
    # Name of the SQL table
    table_name = 'PCIT'
    
    # Path to the output .sql file
    output_sql_file = 'PCIT.sql'
    
    excel_to_sql_file(file_path, table_name, output_sql_file)





