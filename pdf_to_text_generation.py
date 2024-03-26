import os
import base64
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part
import vertexai.preview.generative_models as generative_models
from PyPDF2 import PdfReader, PdfWriter
import time

# Initialize Vertex AI with your project and location
vertexai.init(project="cs4052-assignment2", location="asia-southeast1")

def split_and_generate(source_folder, output_folder, text_folder, model_name="gemini-1.0-pro-vision-001"):
    # Check if the source folder exists
    if not os.path.exists(source_folder):
        print(f"Source folder {source_folder} does not exist.")
        return

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create text folder if it doesn't exist
    if not os.path.exists(text_folder):
         os.makedirs(text_folder)

    # Loop through each file in the source folder
    for filename in os.listdir(source_folder):
        file_without_extension = os.path.splitext(filename)[0]
        # Process only PDF files
        if filename.endswith('.pdf'):
            # Open and read the PDF file
            with open(os.path.join(source_folder, filename), 'rb') as infile:
                pdf_reader = PdfReader(infile)
                num_pages = len(pdf_reader.pages)

                # Create a dedicated output folder for the current PDF
                individual_output_folder = os.path.join(output_folder, filename[:-4])
                if not os.path.exists(individual_output_folder):
                    os.makedirs(individual_output_folder)

                # Split each page of the PDF and write to separate files
                for page in range(num_pages):
                    pdf_writer = PdfWriter()
                    pdf_writer.add_page(pdf_reader.pages[page])
                    output_filename = f"{filename[:-4]}_p{page + 1}.pdf"
                    output_path = os.path.join(individual_output_folder, output_filename)
                    with open(output_path, 'wb') as outfile:
                        pdf_writer.write(outfile)
                    print(f"Created: {output_path}")

                    # Generate content for the current page using the model
                    generate_content(output_path, individual_output_folder, text_folder, model_name, file_without_extension, page + 1)
                    # Wait for 2 seconds before processing the next page
                    time.sleep(2)

def generate_content(pdf_path, output_folder, text_folder, model_name, lecture, page_number):
    # Initialize the generative model with the specified model name
    model = GenerativeModel(model_name)
    # Read the PDF data from the given path
    with open(pdf_path, 'rb') as file:
        pdf_data = file.read()

    # Create a document part with the PDF data
    document = Part.from_data(data=pdf_data, mime_type="application/pdf")
    # Generate content based on the document and predefined instructions
    responses = model.generate_content(
        [document, """
            Given a document, your task is to extract the text from the document
            - If there is a diagram, describe and elaborate the diagram
            - Do not change any content
            - Output should be text only, and not in markdown or any other format
            - Clean up the text such that it is readable and coherent
            """],
        generation_config={
            "max_output_tokens": 2048,
            "temperature": 0,
            "top_p": 1,
            "top_k": 32
        },
        safety_settings={
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        },
        stream=False,
    )
    # Write the generated content to a text file
    try:
        if responses:
            output_file_path = os.path.join(text_folder, f'{lecture}_extracted.txt')
            with open(output_file_path, 'a') as output_file:
                output_file.write(responses.text + "\n")
            print(f"Generated content for page {page_number} added to {output_file_path}")
    except AttributeError as e:
            print(f"Error processing content for page {page_number}: {e}")

# Usage
source_folder = 'lecture/slides'
split_folder = 'lecture/split'
text_folder = 'lecture/text'
model_name = "gemini-1.0-pro-vision-001"
split_and_generate(source_folder, split_folder, text_folder, model_name)
