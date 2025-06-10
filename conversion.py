def parse_extracted_text(text):
    """
    Parse the extracted text into a list of table rows.
    Each row is a dictionary containing the field values.
    """
    try:
        # Split the text into lines and remove empty lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Convert each line into a dictionary
        table_rows = []
        for line in lines:
            try:
                row_data = eval(line)  # Safely evaluate the string as a dictionary
                if isinstance(row_data, dict):
                    table_rows.append(row_data)
            except:
                continue
                
        return table_rows
    except Exception as e:
        print(f"Error parsing extracted text: {e}")
        return []

def build_json_structure(table_rows, project_id, project_name, project_description,
                        file_id, file_name, file_format, scanned_file_name,
                        metadata_id, tabledata_id):
    """
    Build a structured JSON object from the parsed table rows.
    """
    return {
        "project": {
            "id": project_id,
            "name": project_name,
            "description": project_description
        },
        "file": {
            "id": file_id,
            "name": file_name,
            "format": file_format,
            "scanned_file_name": scanned_file_name
        },
        "metadata": {
            "id": metadata_id,
            "tabledata_id": tabledata_id,
            "rows": table_rows
        }
    } 