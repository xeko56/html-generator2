import random
import os
import json
import numpy as np
from bs4 import BeautifulSoup
from openai import OpenAI
from html2image import Html2Image

os.environ["OPENAI_API_KEY"] = "API_KEY"

def should_merge(percentage):
    """
    Determine if a cell should be merged based on a given percentage.
    
    Args:
    
        percentage (int): The percentage chance that a cell should be merged.
        
        Returns:
            bool: True if the cell should be merged, False otherwise    
    """
    return random.random() < percentage / 100

def create_cell(soup, tag, styles, content="", rowspan=1, colspan=1):
    cell = soup.new_tag(tag, style=styles)
    if rowspan > 1:
        cell['rowspan'] = rowspan
    if colspan > 1:
        cell['colspan'] = colspan
    cell.string = content
    return cell

def generate_html_table(header_merge_percentage, body_merge_percentage):
    soup = BeautifulSoup("", "html.parser")
    table = soup.new_tag("table", style="border-collapse: collapse;")

    rows = np.random.randint(3, 15)
    cols = np.random.randint(2, 10)       
    
    # Initialize grid for tracking cell spans
    grid = [[(1, 1) for _ in range(cols)] for _ in range(rows)]

    # Function to check if a span is possible
    def is_span_possible(row, col, rowspan, colspan):
        if row + rowspan > rows or col + colspan > cols:
            return False
        for i in range(row, row + rowspan):
            for j in range(col, col + colspan):
                if grid[i][j] != (1, 1):  # Span not possible if any cell in the span is already taken
                    return False
        return True

    for i in range(rows):
        j = 0
        while j < cols:
            current_percentage = header_merge_percentage if i == 0 else body_merge_percentage
            if should_merge(current_percentage):
                rowspan = random.randint(1, rows - i)
                colspan = random.randint(1, cols - j)

                if is_span_possible(i, j, rowspan, colspan):
                    # Apply span
                    for m in range(rowspan):
                        for n in range(colspan):
                            if m > 0 or n > 0:
                                grid[i + m][j + n] = (0, 0)  # Mark cells within the span
                    grid[i][j] = (rowspan, colspan)
                    j += colspan - 1  # Skip cells that are already included in the colspan
            j += 1

    for i in range(rows):
        tr = soup.new_tag("tr")
        for j in range(cols):
            if grid[i][j] != (0, 0):  # Only create cells that start spans
                rowspan, colspan = grid[i][j]
                cell_tag = "th" if i == 0 else "td"
                cell = create_cell(soup, cell_tag, "border: 1px solid black;", f"Cell ({i},{j})", rowspan, colspan)
                tr.append(cell)
        table.append(tr)

    return str(table)

def populate_content(html_table):
    """
    Populate the HTML table with semantic content using OpenAI's GPT-4o-mini model.
    
    Args:
        html_table (str): The HTML table to populate with content.
        
        Returns:
            str: The HTML table with populated content.     
    """
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "I have an HTML table with specific dimensions, rowspan, colspan and dummy data and I need to populate it with meaningful content. The table should be filled in a way that each cell has unique and appropriate content related to the table's context. The table has header cells, and the remaining cells should contain data without strange unicode symbols, please don't remove any table cell on purpose."},
            {"role": "user", "content": html_table}
        ]
    )
    content = response.choices[0].message.content
    soup = BeautifulSoup(content, 'html.parser')
    table = soup.find('table')

    if table:
        html_content = str(table)
    else:
        raise ValueError("Table not found")
    return html_content

def save_html_to_file(html_content, filename):
    """
    Save HTML content to a file.
    
    Args:
        html_content (str): HTML content to save.
        filename (str): The file path where the HTML content will be saved.
    """
    with open(filename, 'w') as file:
        file.write(html_content)

def render_html_to_image(html_file, output_file):
    """
    Render an HTML file to an image using Html2Image.
    
    Args:
        html_file (str): The HTML file path to render.
        output_file (str): The output image file path.
    """
    hti = Html2Image()
    hti.output_path = 'images'
    hti.screenshot(html_file=html_file, save_as=output_file)            

def main():
    # Configuration
    body_merge_percentage = 1  # 1% chance to merge cells
    header_merge_percentage = 20  # 20% chance to merge header cells

    os.makedirs('tables', exist_ok=True)
    os.makedirs('images', exist_ok=True)
    images_folder = './images'
    html_folder = './tables'
    output_file = 'data_pairs.json'    

    # Generate 50 HTML tables
    for i in range(100):
        empty_table = generate_html_table(header_merge_percentage, body_merge_percentage)
        html_table = populate_content(empty_table)
        debug_filename = f'tables/debug_{i}.html'
        html_filename = f'tables/table_{i}.html'
        save_html_to_file(empty_table, debug_filename)
        save_html_to_file(html_table, html_filename)
        render_html_to_image(html_filename, f'table_{i}.png')

    data_pairs = []

    # Iterate over all files in the images folder
    for image_filename in os.listdir(images_folder):
        if image_filename.endswith('.png'):
            base_filename = os.path.splitext(image_filename)[0]
            html_filename = base_filename + '.html'
            
            image_path = os.path.join(images_folder, image_filename)
            html_path = os.path.join(html_folder, html_filename)
            
            if os.path.exists(html_path):
                with open(html_path, 'r') as html_file:
                    html_content = html_file.read()
                    
                data_pairs.append({"image": image_path, "html": html_content})

    # Save the data pairs to a JSON file
    with open(output_file, 'w') as f:
        json.dump(data_pairs, f, indent=4)

    print(f"Data pairs saved to {output_file}")    

if __name__ == "__main__":
    main()
