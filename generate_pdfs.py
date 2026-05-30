import os
from fpdf import FPDF
from fpdf.enums import XPos, YPos

class USTPDF(FPDF):
    def header(self) -> None:
        self.set_font('Helvetica', 'B', 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, 'UST AUTOMOTIVE DATA SCIENCE INTERNSHIP DRIVE - JUNE 2026', 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='R')
        self.set_draw_color(200, 200, 200)
        self.line(10, 18, 200, 18)
        self.ln(5)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(120, 120, 120)
        self.set_draw_color(200, 200, 200)
        self.line(10, self.h - 18, 200, self.h - 18)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')

def clean_latin1(text: str) -> str:
    """
    Cleans Unicode characters to fit Latin-1 encoding used by FPDF core fonts.
    Replaces box drawing, math symbols, and arrows with standard equivalents.
    """
    replacements = {
        '│': '|',
        '─': '-',
        '┼': '+',
        '▼': 'v',
        '▲': '^',
        '►': '>',
        '◄': '<',
        '•': '*',
        '→': '->',
        '←': '<-',
        '½': '1/2',
        '²': '^2',
        '³': '^3',
        'Δ': 'Delta',
        '∆': 'Delta',
        'θ': 'theta',
        'π': 'pi',
        'σ': 'sigma',
        '±': '+/-',
        '≤': '<=',
        '≥': '>=',
        '∞': 'inf',
        '≈': '~'
    }
    for orig, rep in replacements.items():
        text = text.replace(orig, rep)
    # Safely strip remaining non-latin1 characters
    return text.encode('latin-1', 'ignore').decode('latin-1')

def create_title_page(pdf: USTPDF, title: str, subtitle: str) -> None:
    pdf.add_page()
    pdf.ln(50)
    pdf.set_font('Helvetica', 'B', 22)
    pdf.set_text_color(30, 41, 59)  # Dark slate
    pdf.multi_cell(0, 12, clean_latin1(title), align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(10)
    
    pdf.set_font('Helvetica', 'I', 14)
    pdf.set_text_color(100, 116, 139)  # Slate grey
    pdf.multi_cell(0, 8, clean_latin1(subtitle), align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(60)
    
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 8, 'Candidate Name: Sanchu Srijan', 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    
    pdf.set_font('Helvetica', '', 11)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(0, 8, 'Assignment: Problem Statement #4 (Coding)', 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    pdf.cell(0, 8, 'Focus: Reinforcement Learning-Based Human Driver Behaviour Modelling', 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    pdf.cell(0, 8, 'Target Platform: Eclipse OpenPASS via FMI 2.0 (OSMP Bridge)', 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    pdf.cell(0, 8, 'Date: May 2026', 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')

def parse_markdown_to_pdf(pdf: USTPDF, filename: str) -> None:
    if not os.path.exists(filename):
        print(f"Warning: File {filename} not found.")
        return
        
    with open(filename, 'r') as f:
        lines = f.readlines()
        
    in_code_block = False
    
    for line in lines:
        stripped = line.strip()
        
        # Handle code blocks
        if stripped.startswith('```'):
            in_code_block = not in_code_block
            pdf.ln(2)
            continue
            
        if in_code_block:
            pdf.set_x(15)  # Indent code block
            pdf.set_font('Courier', '', 8.5)
            pdf.set_text_color(51, 65, 85)
            clean_text = clean_latin1(line.replace('\t', '    ').replace('\n', ''))
            pdf.cell(0, 4, clean_text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            continue
            
        # Handle Headings
        if stripped.startswith('# '):
            pdf.ln(8)
            pdf.set_font('Helvetica', 'B', 16)
            pdf.set_text_color(30, 41, 59)
            pdf.cell(0, 10, clean_latin1(stripped[2:]), 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(2)
        elif stripped.startswith('## '):
            pdf.ln(6)
            pdf.set_font('Helvetica', 'B', 13)
            pdf.set_text_color(30, 41, 59)
            pdf.cell(0, 8, clean_latin1(stripped[3:]), 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(2)
        elif stripped.startswith('### '):
            pdf.ln(4)
            pdf.set_font('Helvetica', 'B', 10.5)
            pdf.set_text_color(71, 85, 105)
            pdf.cell(0, 6, clean_latin1(stripped[4:]), 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(1)
        # Bullet points
        elif stripped.startswith('- ') or stripped.startswith('* '):
            pdf.set_x(15)  # Indent bullet points
            pdf.set_font('Helvetica', '', 10)
            pdf.set_text_color(51, 65, 85)
            text = line.replace('- ', '').replace('* ', '').strip()
            pdf.multi_cell(0, 5, '* ' + clean_latin1(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        # Horizontal rules
        elif stripped == '---':
            pdf.ln(4)
            pdf.set_draw_color(226, 232, 240)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(4)
        # Blank lines
        elif not stripped:
            pdf.ln(2)
        # Normal paragraphs
        else:
            pdf.set_font('Helvetica', '', 10)
            pdf.set_text_color(51, 65, 85)
            text = stripped.replace('**', '').replace('`', '').replace('$$', '').replace('$', '')
            pdf.multi_cell(0, 5, clean_latin1(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

def add_code_file_to_pdf(pdf: USTPDF, filepath: str, title: str) -> None:
    if not os.path.exists(filepath):
        print(f"Warning: File {filepath} not found.")
        return
        
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 10, clean_latin1(title), 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('Helvetica', 'I', 9)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 5, f"Location: {filepath}", 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    
    with open(filepath, 'r') as f:
        code_lines = f.readlines()
        
    pdf.set_font('Courier', '', 8)
    pdf.set_text_color(51, 65, 85)
    
    # Print code lines
    for i, line in enumerate(code_lines, 1):
        clean_line = line.replace('\t', '    ').replace('\n', '')
        formatted_line = f"{i:3d} | {clean_line}"
        pdf.multi_cell(0, 3.5, clean_latin1(formatted_line), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

def create_resume_pdf(pdf: USTPDF) -> None:
    pdf.add_page()
    pdf.ln(10)
    
    # Name
    pdf.set_font('Helvetica', 'B', 22)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 10, 'SANCHU SRIJAN', 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    
    # Contact
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 5, 'Email: sanchusrijan@gmail.com | GitHub: github.com/sanchusrijan', 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    pdf.ln(5)
    
    pdf.set_draw_color(226, 232, 240)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)
    
    # Summary
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 6, 'PROFESSIONAL SUMMARY', 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(71, 85, 105)
    pdf.multi_cell(0, 5, 'Data Science and Machine Learning enthusiast with project experience in Deep Reinforcement Learning, autonomous driving simulation, and control logic design. Specialized in training reactive agent models and exporting standards-compliant co-simulation Functional Mock-up Units (FMUs).', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)
    
    # Education
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 6, 'EDUCATION', 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 5, 'B.Tech / M.Sc in Data Science / Computer Science', 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('Helvetica', '', 9.5)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 5, 'Specialization: Machine Learning & Intelligent Vehicles | Expected Graduation: 2027', 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)
    
    # Technical Skills
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 6, 'TECHNICAL SKILLS', 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(71, 85, 105)
    pdf.multi_cell(0, 5, '* Programming: Python (3.11+), SQL, C++\n* Reinforcement Learning: PyTorch, Gymnasium, Stable-Baselines3 (PPO, DQN, DDPG)\n* Simulation Frameworks: MetaDrive driving simulator, OpenPASS integration\n* Automotive Standards: ASAM OpenSCENARIO XML format, Functional Mock-up Interface FMI 2.0 co-simulation\n* Exporters/Tools: pythonfmu, fmpy, Git, Markdown', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)
    
    # Projects
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 6, 'ACADEMIC PROJECTS', 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 5, 'Reinforcement Learning-Based Human Driver Behaviour Simulation (UST Drive)', 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(71, 85, 105)
    pdf.multi_cell(0, 5, '* Wrapped MetaDrive simulator in a custom Gymnasium environment featuring 259 lidar-state observations.\n* Authored multi-objective reward functions balancing safety cushion, headway, and jerk acceleration metrics.\n* Trained actor-critic models for Cautious and Aggressive driver profiles using SB3 PPO.\n* Exported policy models to FMI 2.0 co-simulation slaves (FMU format) and validated FMI variables with FMPy.\n* Modeled five test scenarios (S1-S5) utilizing ASAM OpenSCENARIO XML structures.', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(5)

def main() -> None:
    print("Compiling Approach Document PDF...")
    pdf_approach = USTPDF()
    pdf_approach.alias_nb_pages()
    create_title_page(pdf_approach, "TECHNICAL APPROACH DOCUMENT", "System Design, Reward Formulations & Validation Rationale")
    pdf_approach.add_page()
    
    parse_markdown_to_pdf(pdf_approach, "documents/approach_document.md")
    pdf_approach.add_page()
    parse_markdown_to_pdf(pdf_approach, "documents/reward_design.md")
    
    pdf_approach.output("SanchuSrijan-Approach.pdf")
    print("Successfully generated SanchuSrijan-Approach.pdf")
    
    print("\nCompiling Code Document PDF...")
    pdf_code = USTPDF()
    pdf_code.alias_nb_pages()
    create_title_page(pdf_code, "SOURCE CODE REFERENCE", "Complete implementation files with Python type-hinting")
    
    add_code_file_to_pdf(pdf_code, "src/environment.py", "1. Custom Driving Environment Wrapper")
    add_code_file_to_pdf(pdf_code, "src/train.py", "2. Policy Network Training Pipeline")
    add_code_file_to_pdf(pdf_code, "src/evaluate.py", "3. Evaluation & KPI Validation Runner")
    add_code_file_to_pdf(pdf_code, "src/export_fmu.py", "4. FMU 2.0 Co-Simulation Slave Exporter")
    
    pdf_code.output("SanchuSrijan-Code.pdf")
    print("Successfully generated SanchuSrijan-Code.pdf")
    
    print("\nCompiling Resume Document PDF...")
    pdf_resume = USTPDF()
    pdf_resume.alias_nb_pages()
    create_resume_pdf(pdf_resume)
    pdf_resume.output("SanchuSrijan-Resume.pdf")
    print("Successfully generated SanchuSrijan-Resume.pdf")

if __name__ == "__main__":
    main()
