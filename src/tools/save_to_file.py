from pydantic import BaseModel, Field
from langchain.tools import StructuredTool


class SaveToolInput(BaseModel):
    content: str = Field(..., description="The content to save")
    filename: str = Field(...,
                          description="The name of the file to save the content to")


def save_to_file(content: str, filename: str = 'content.txt'):
    """
    Save the given content to a text file.

    Args:
        content (str): The content to save.
        filename (str): The name of the file to save the content to.
    """

    with open(f'output/{filename}', 'w') as file:
        file.write(content)
    return f"Content saved to {filename}"


save_tool = StructuredTool.from_function(
    func=save_to_file,
    name="save_to_file",
    description="Saves the provided content to a file, input should be the content",
    args_schema=SaveToolInput
)
