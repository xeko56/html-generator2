import random
import os
import json
import pathlib
import tinycss2
import random
import numpy as np
from PIL import Image
from bs4 import BeautifulSoup
from openai import OpenAI
from html2image import Html2Image
from huggingface_hub import HfApi

os.environ["OPENAI_API_KEY"] = "sk-proj-qCuIGSjvbNnTrJQl6wHTT3BlbkFJGEFGiYqRCE0u4ehGDyes"

def create_huggingface_repo(repo_name, organization_name="username", private=False):
    """
    Create a new Hugging Face repository.
    
    Args:
        repo_name (str): The name of the repository to create.
        organization_name (str): The name of the organization to create the repository under.
        private (bool): Whether the repository should be private.
        
    Returns:
        dict: The response from the Hugging Face API.
    """
    api = HfApi()
    return api.create_repo(repo_name, organization=organization_name, private=private)

def upload_files_to_huggingface_repo(repo_id, files):
    """
    Upload files to a Hugging Face repository.
    
    Args:
        repo_id (str): The ID of the repository to upload files to.
        files (list): A list of dictionaries containing the file name and file path.
        
    Returns:
        dict: The response from the Hugging Face API.
    """
    api = HfApi()
    return api.upload_files(repo_id, files)

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
    cell = soup.new_tag(tag)
    if rowspan > 1:
        cell['rowspan'] = rowspan
    if colspan > 1:
        cell['colspan'] = colspan
    cell.string = content
    return cell

def generate_html_table(header_merge_percentage, body_merge_percentage):
    soup = BeautifulSoup("", "html.parser")
    table = soup.new_tag("table")

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

def clean_html_structure(html_table):
    """
    Cleans the HTML structure by removing the content within the table cells,
    leaving only the tags, attributes, and structure.
    
    Args:
        html_table (str): The HTML table to clean.
        
    Returns:
        str: The cleaned HTML structure.
    """
    soup = BeautifulSoup(html_table, 'html.parser')
    
    # Remove content within tags but keep the structure
    for cell in soup.find_all(['td', 'th']):
        cell.clear()
    
    # Return the cleaned HTML structure as a string
    return str(soup)

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
            {
                "role": "system", 
                "content": "I have an HTML table with specific dimensions, each row and column meticulously planned to suit the layout's needs. It features cells with rowspan and colspan where specified. Please populate this table with content relevant to a conference schedule, adhering strictly to the existing structure. Ensure no cells are added or removed, and the rowspan and colspan attributes are respected to maintain the layout integrity."},
            {
                "role": "user", 
                "content": html_table
            }
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

def extract_css_classes(css: str):
    """Extracts class selectors and their styles from a CSS string."""
    rules = tinycss2.parse_stylesheet(css, skip_comments=True, skip_whitespace=True)
    class_definitions = []
    for rule in rules:
        if rule.type == "qualified-rule":
            prelude = tinycss2.serialize(rule.prelude).strip()
            # print(prelude)
            if prelude.startswith('table.'):
                declarations = tinycss2.parse_declaration_list(rule.content, skip_comments=True, skip_whitespace=True)
                styles = []
                for decl in declarations:
                    if decl.type == "declaration":
                        prop = decl.name
                        value = tinycss2.serialize(decl.value).strip()
                        styles.append(f"{prop}: {value};")
                class_definitions.append((prelude, styles))

    return class_definitions

def extract_class_names(css):
    rules = tinycss2.parse_stylesheet(css, skip_comments=True, skip_whitespace=True)

    for rule in rules:
        if rule.type == "qualified-rule":
            prelude = tinycss2.serialize(rule.prelude).strip()
            if prelude.startswith('table.'):
                # Extract the class name that follows 'table.'
                class_name = prelude.split(' ')[0].split('.')[1]
                return class_name
    return None


def save_html_to_file(html_content, filename):
    """
    Save HTML content to a file.
    
    Args:
        html_content (str): HTML content to save.
        filename (str): The file path where the HTML content will be saved.
    """
    with open(filename, 'w', encoding='utf-8') as file:
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

def get_bounding_box(html_file):
    """
    Get the bounding box of the HTML content in an image.
    
    Args:
        html_file (str): The HTML file path to render.
        
    Returns:
        dict: The bounding box coordinates of the HTML content in the image.
    """
    image = Image.open(html_file)
    bbox = image.getbbox()

    cropped_image = image.crop(bbox)

    # Save or display the cropped image
    cropped_image.save(html_file)

def main():
    # Configuration
    body_merge_percentage = 1  # 1% chance to merge cells
    header_merge_percentage = 20  # 20% chance to merge header cells

    os.makedirs('tables', exist_ok=True)
    os.makedirs('images', exist_ok=True)
    images_folder = './images'
    html_folder = './tables'
    output_file = 'data_pairs.json'    

    # Generate HTML and CSS
    # for i in range(100):
    for i in range(201, 800):
        empty_table = generate_html_table(header_merge_percentage, body_merge_percentage)
        base_html_table = populate_content(empty_table)
        debug_filename = f'tables/debug_{i}.html'
        base_html_filename = f'tables/base_table_{i}.html'
        save_html_to_file(empty_table, debug_filename)
        save_html_to_file(base_html_table, base_html_filename)

        # Put CSS Style to base table
        html_table = add_css_to_html(i)
        html_filename = f'tables/table_{i}.html'
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
                with open(html_path, 'r', encoding='utf-8') as html_file:
                    html_content = html_file.read()
                    
                data_pairs.append({"image": image_path, "html": html_content})

    # Save the data pairs to a JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data_pairs, f, indent=4)

    print(f"Data pairs saved to {output_file}")

def add_css_to_html(idx):
    css_path = pathlib.Path('styles') / f'style_{idx}.css'
    css_content = css_path.read_text()

    existing_html_path = pathlib.Path('tables') / f'base_table_{idx}.html'
    existing_html = existing_html_path.read_text(encoding='utf-8')
    
    soup = BeautifulSoup(existing_html, 'html.parser')

    if not soup.html:
        html_tag = soup.new_tag('html')
        soup.insert(0, html_tag)
        if not soup.head:
            head_tag = soup.new_tag('head')
            html_tag.append(head_tag)
        if not soup.body:
            body_tag = soup.new_tag('body')
            html_tag.append(body_tag)

    table = soup.find('table')

    class_name = extract_class_names(css_content)

    if table and class_name:
        # Add the class if it doesn't already exist
        existing_classes = table.get('class', [])
        if class_name not in existing_classes:
            table['class'] = existing_classes + [class_name]

    head = soup.head
    style_tag = soup.new_tag('style')
    style_tag.string = css_content
    head.append(style_tag)

    # Double check
    for element in soup.contents:
        if element.name not in ['html', 'head', 'body']:
            if soup.body:
                soup.body.append(element.extract())
            else:
                body_tag = soup.new_tag('body')
                soup.html.append(body_tag)
                body_tag.append(element.extract())             

    return str(soup)

def huggingface_main():
    # Create a new Hugging Face repository
    repo_name = "html-tables"
    repo = create_huggingface_repo(repo_name, organization_name="xeko56", private=False)
    repo_id = repo["id"]
    print(f"Created Hugging Face repository: {repo_name}")

    # Upload the data pairs to the Hugging Face repository
    output_file = 'data_pairs.json'
    files = [{"filename": output_file, "filepath": output_file}]
    upload_files_to_huggingface_repo(repo_id, files)
    print(f"Uploaded data pairs to Hugging Face repository: {repo_name}")

def test():    
    for i in range(1, 801):
        image_file =  f'images/table_{i}.png'
        get_bounding_box(image_file)

if __name__ == "__main__":
    test()
    # main()
    # add_css_to_html(15)
