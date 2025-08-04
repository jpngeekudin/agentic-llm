from pydantic import BaseModel, Field
from langchain.tools import StructuredTool
import os


class SaveToolInput(BaseModel):
    content: str = Field(..., description="The content to save")
    project: str = Field(...,
                         description="Projet name, where will be the root directory")
    path: str = Field(...,
                      description="Relative path of the file in the project directory")
    filename: str = Field(...,
                          description="The name of the file to save the content to")


def save_to_project(content: str, project: str, path: str, filename: str = 'content.txt'):
    """
    Save content into a file with specified directory for creating structured project folders.

    Args:
        content (str): The content to save.
        project (str): Project name, where will be the root directory.
        path (str): Relative path of the file in the project directory.
        filename (str): The name of the file to save the content to.
    """

    full_path = os.path.join('./output', project, path, filename)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, 'w') as file:
        file.write(content)
    return f"Content saved to {full_path}"


save_project_tool = StructuredTool.from_function(
    func=save_to_project,
    name="save_to_project",
    description="Save content into a file with specified directory for creating structured project folders. The parameters content, project, path, and filename are required.",
    args_schema=SaveToolInput
)
