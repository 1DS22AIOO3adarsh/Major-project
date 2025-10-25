import os
import shutil
import uuid # To create unique folder names
import subprocess # Import the subprocess module
import platform # To check the operating system

print("🚀 Initializing paper assembly and compilation script...")

# --- Step 1: Simulate User Input --- (Keep this the same)
paper_content = {
    "title": "My Automatically Formatted Paper",
    "author_info": """
\\author{
    \\IEEEauthorblockN{Your Name}
    \\IEEEauthorblockA{\\textit{Your Department} \\\\
    \\textit{Your University}\\\\
    City, Country \\\\
    email@example.com}
}""",
    "abstract": "This is the abstract text provided by the user.",
    "keywords": "formatting, automation, LaTeX, IEEE",
    "introduction": "This is the introduction section.",
    "methodology": "Here, the user describes the methodology.",
    "results": "The results section contains the findings.",
    "conclusion": "Finally, the conclusion summarizes the work.",
    "bibtex_references": """
@article{example_ref,
  title={An Example Reference},
  author={Doe, John},
  journal={Journal of Examples},
  volume={1},
  pages={1--10},
  year={2025}
}"""
}

# --- Step 2: Define the Assembly Function --- (Keep this the same)
def assemble_latex_project(content, output_folder_name="output_paper"):
    project_path = os.path.join(os.getcwd(), output_folder_name)
    figures_path = os.path.join(project_path, 'figures')

    if os.path.exists(project_path):
        shutil.rmtree(project_path)

    os.makedirs(figures_path, exist_ok=True)
    print(f"✅ Created output folder at: {project_path}")

    default_intro = '\\lipsum[1]'
    default_method = '\\lipsum[2]'
    default_results = '\\lipsum[3]'
    default_conclusion = '\\lipsum[4]'

# ... inside assemble_latex_project function ...

    # --- Step 3: Define the LaTeX Template ---
    latex_template = f"""
\\documentclass[conference]{{IEEEtran}}

% ... (Packages remain the same) ...
\\usepackage{{lipsum}} % For placeholder text

\\title{{{content.get('title', 'Default Title')}}}
{content.get('author_info', '')}
\\begin{{document}}
\\maketitle
\\begin{{abstract}}
{content.get('abstract', '')}
\\end{{abstract}}
\\begin{{IEEEkeywords}}
{content.get('keywords', 'default, keywords')}
\\end{{IEEEkeywords}}

\\section{{Introduction}}
{content.get('introduction', default_intro)} 
% --- ADD A CITATION HERE ---
This is where we cite our example reference \\cite{{example_ref}}. 
% --- END ADDITION ---

\\section{{Methodology}}
{content.get('methodology', default_method)}

\\section{{Results and Discussion}}
{content.get('results', default_results)}

\\section{{Conclusion}}
{content.get('conclusion', default_conclusion)}

% --- Bibliography ---
\\bibliographystyle{{IEEEtran}}
\\bibliography{{references}}

\\end{{document}}
"""
# ... (rest of the script remains the same) ...

    main_tex_path = os.path.join(project_path, 'main.tex')
    with open(main_tex_path, 'w', encoding='utf-8') as f:
        f.write(latex_template)
    print(f"   - Created {os.path.basename(main_tex_path)}")

    references_bib_path = os.path.join(project_path, 'references.bib')
    with open(references_bib_path, 'w', encoding='utf-8') as f:
        f.write(content.get('bibtex_references', ''))
    print(f"   - Created {os.path.basename(references_bib_path)}")

    print("✅ Assembly complete!")
    return project_path

# --- Step 3: Define the Compilation Function --- (NEW)
def compile_latex_to_pdf(project_dir):
    """Compiles the main.tex file in the project directory to PDF."""
    main_tex_file = "main.tex"
    main_base_name = "main" # without .tex extension

    # Determine the command based on OS
    pdflatex_cmd = "pdflatex"
    bibtex_cmd = "bibtex"
    if platform.system() == "Windows":
         # Commands might be just the name if PATH is set correctly,
         # otherwise, you might need the full path if errors occur.
         pass # Use default command names

    print(f"⚙️  Starting compilation in '{project_dir}'...")

    # Command sequence for compilation with bibliography
    commands = [
        [pdflatex_cmd, "-interaction=nonstopmode", main_tex_file],
        [bibtex_cmd, main_base_name],
        [pdflatex_cmd, "-interaction=nonstopmode", main_tex_file],
        [pdflatex_cmd, "-interaction=nonstopmode", main_tex_file]
    ]

    compilation_successful = True
    for cmd in commands:
        try:
            # Run the command silently, capture output
            print(f"   - Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=project_dir, check=True, capture_output=True, text=True)
            # You can check result.stdout for detailed logs if needed
        except FileNotFoundError:
            print(f"   - ❌ Error: Command '{cmd[0]}' not found. Is LaTeX installed and in PATH?")
            compilation_successful = False
            break
        except subprocess.CalledProcessError as e:
            print(f"   - ❌ Error during compilation ({' '.join(cmd)}):")
            print(e.stdout) # Show LaTeX errors
            print(e.stderr)
            compilation_successful = False
            break
        except Exception as e:
            print(f"   - ❌ An unexpected error occurred: {e}")
            compilation_successful = False
            break

    if compilation_successful:
        pdf_path = os.path.join(project_dir, f"{main_base_name}.pdf")
        if os.path.exists(pdf_path):
            print(f"✅ Compilation successful! PDF created at: {pdf_path}")
            return pdf_path
        else:
             print(f"   - ⚠️ Warning: Compilation seemed to succeed, but PDF file not found.")
             return None
    else:
        print("❌ Compilation failed.")
        return None

# --- Step 4: Run Assembly and Compilation --- (Updated)
output_dir = assemble_latex_project(paper_content)
if output_dir:
    compile_latex_to_pdf(output_dir)