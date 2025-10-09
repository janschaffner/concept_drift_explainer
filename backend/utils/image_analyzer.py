"""
This module provides a utility for analyzing the content of images using a
multimodal Large Language Model (GPT-4o with vision).

Its primary purpose is to extract structured, textual information from visual
media such as organizational charts, process diagrams, and slides. The generated
text can then be ingested into the vector database, making the visual content
searchable and available as context for the drift explanation pipeline.
"""

import base64
import logging
from pathlib import Path
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# --- Path Correction & Configuration ---
project_root = Path(__file__).resolve().parents[2]
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Vision capabilities require the more powerful models like GPT-4o
MODEL_NAME = "gpt-4o" 

def encode_image(image_path: Path) -> str:
    """Reads an image file and encodes it into a base64 string.

    This is a necessary preprocessing step to prepare an image to be sent to the
    OpenAI API in a JSON payload.

    Args:
        image_path: The path to the image file.

    Returns:
        A base64 encoded string representation of the image, or an empty
        string if an error occurs during encoding.
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logging.error(f"Error encoding image {image_path}: {e}")
        return ""

def analyze_image_content(image_path: Path) -> str:
    """
    Analyzes an image using a multimodal LLM and returns a text description.

    This function orchestrates the image analysis process:
    1. Encodes the image into a base64 string.
    2. Constructs a detailed, role-specific prompt instructing the LLM to act
       as a business analyst.
    3. Sends the prompt and image data to the GPT-4o vision model.
    4. Returns the LLM's textual description of the image content.

    Args:
        image_path: The path to the image file to be analyzed.

    Returns:
        A string containing the LLM-generated description of the image, or an
        error message if the process fails.
    """
    logging.info(f"Analyzing image content for: {image_path.name}")
    
    # Step 1: Encode the image to a base64 string for the API payload.
    base64_image = encode_image(image_path)
    if not base64_image:
        return "Error encoding image."

    # Determine the correct MIME type (e.g., 'image/png') from the file extension.
    mime_type = f"image/{image_path.suffix.lstrip('.')}"
    
    # Initialize the vision-capable LLM.
    llm = ChatOpenAI(model=MODEL_NAME, max_tokens=1024)
    
    # Step 2: Construct the multimodal prompt. This is a list containing a single
    # HumanMessage object. The content of this message is itself a list, with
    # one dictionary for the text instructions and another for the image data.
    prompt = [
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": """You are an expert business analyst. Analyze the following image and provide a detailed, factual description of its content.
                    - If it is a chart or graph, describe what it shows, including its title, axes, and the data trends.
                    - If it is an organizational chart, describe the reporting structure, roles, and departments shown.
                    - If it contains text, transcribe the text accurately.
                    Your description will be used as context to explain a business process change, so focus on information relevant to that goal."""
                },
                {
                    "type": "image_url",
                    # The image is passed as a data URL with the base64-encoded string.
                    "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}
                }
            ]
        )
    ]

    # Step 3: Invoke the LLM and return the generated description.
    try:
        # Invoke the LLM with the multimodal prompt.
        response = llm.invoke(prompt)
        description = response.content
        logging.info(f"Successfully generated description for {image_path.name}")
        return description
    except Exception as e:
        logging.error(f"Error analyzing image with LLM: {e}")
        return f"Error analyzing image: {e}"
    

# This block allows the script to be run directly for testing purposes.
if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    
    # To test, place an image named 'test_org_chart.png' in the 'data/documents/' folder.
    test_image_path = project_root / "data" / "documents" / "2025-07-07_empireorg.png"
    if test_image_path.exists():
        description = analyze_image_content(test_image_path)
        print("\n--- Generated Description ---")
        print(description)
    else:
        print(f"Test image not found at {test_image_path}. Please place an image there to test.")