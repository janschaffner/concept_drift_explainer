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
    """
    Reads an image file from the given path and encodes it into a base64 string.

    Args:
        image_path: The path to the image file.

    Returns:
        A base64 encoded string representation of the image, or an empty string on error.
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logging.error(f"Error encoding image {image_path}: {e}")
        return ""

def analyze_image_content(image_path: Path) -> str:
    """
    Analyzes an image using a multimodal LLM (GPT-4o with vision) and returns a
    detailed text description of its content.

    This function first encodes the image into base64, then constructs a prompt
    that instructs the LLM to act as a business analyst and describe the image's
    content, focusing on aspects relevant to process analysis (e.g., org charts,
    graphs, or transcribed text).

    Args:
        image_path: The path to the image file to be analyzed.

    Returns:
        A string containing the LLM-generated description of the image.
    """
    logging.info(f"Analyzing image content for: {image_path.name}")
    
    base64_image = encode_image(image_path)
    if not base64_image:
        return "Error encoding image."

    # Determine the correct MIME type (e.g., 'image/png') from the file extension.
    mime_type = f"image/{image_path.suffix.lstrip('.')}"
    
    # Initialize the vision-capable LLM.
    llm = ChatOpenAI(model=MODEL_NAME, max_tokens=1024)
    
    # The prompt is a list containing a HumanMessage object. This object's content
    # is a list of dictionaries, one for the text instruction and one for the image data.
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

    try:
        # Invoke the LLM with the multimodal prompt.
        response = llm.invoke(prompt)
        description = response.content
        logging.info(f"Successfully generated description for {image_path.name}")
        return description
    except Exception as e:
        logging.error(f"Error analyzing image with LLM: {e}")
        return f"Error analyzing image: {e}"

if __name__ == '__main__':
    # This block allows the script to be run directly for testing purposes.
    from dotenv import load_dotenv
    load_dotenv()
    
    # To test, place an image named 'test_org_chart.png' in the 'data/documents/' folder.
    test_image_path = project_root / "data" / "documents" / "test_org_chart.png"
    if test_image_path.exists():
        description = analyze_image_content(test_image_path)
        print("\n--- Generated Description ---")
        print(description)
    else:
        print(f"Test image not found at {test_image_path}. Please place an image there to test.")