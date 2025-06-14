# Leadership Insights with Prof. Gupta 💡

A Streamlit-based chatbot application that allows students to interact with Professor Vishal Gupta's Leadership Skills course content through an AI-powered conversational interface. This tool leverages the power of Large Language Models (LLMs) and Retrieval Augmented Generation (RAG) to provide accurate and context-aware answers.

## ✨ Features

- **Interactive Chat Interface**: Ask questions about leadership concepts and get responses based on Professor Gupta's course transcripts.
- **AI-Powered Responses**: Utilizes OpenAI's `gpt-4o` model integrated with LlamaIndex for intelligent document retrieval and generation.
- **Course-Specific Knowledge**: Ensures responses are exclusively derived from Professor Gupta's Leadership Skills course materials, providing focused and relevant information.
- **Real-time Processing**: Employs asynchronous operations for a smooth and responsive user experience.
- **Secure Configuration**: API keys and other sensitive information are managed securely using Streamlit's secrets management (`.streamlit/secrets.toml`).
- **Easy Setup**: Clear instructions for installation and configuration.

## 📋 Prerequisites

Before you begin, ensure you have the following:

- Python 3.8 or higher
- An active OpenAI API key
- A LlamaCloud API key and associated LlamaCloud Project Name, Index Name, and Organization ID
- `pip` for installing Python packages
- `git` for cloning the repository (if you don't have the files already)

## 🛠️ Installation

Follow these steps to get the application running on your local machine:

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone https://github.com/your-username/leadership-chatbot.git
    cd leadership-chatbot
    ```
    *(Replace `https://github.com/your-username/leadership-chatbot.git` with the actual repository URL if applicable)*

2.  **Navigate to the project directory:**
    If you've cloned the repository, you should already be in the directory. If you downloaded the files, navigate to that directory.
    ```bash
    cd path/to/leadership-chatbot
    ```

3.  **Create and activate a virtual environment (recommended):**
    This helps manage project dependencies.

    *   On Windows:
        ```bash
        python -m venv venv
        venv\Scripts\activate
        ```
    *   On macOS/Linux:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

4.  **Install dependencies:**
    All required Python packages are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

## ⚙️ Configuration

The application requires API keys and LlamaCloud details to function. These are stored in a `secrets.toml` file within a `.streamlit` directory.

1.  **Create the secrets file:**
    In the root directory of the project, create a folder named `.streamlit` if it doesn't already exist. Inside this folder, create a file named `secrets.toml`.

    Your project structure should look something like this:
    ```
    leadership-chatbot/
    ├── .streamlit/
    │   └── secrets.toml
    ├── app.py
    ├── requirements.txt
    └── ... (other project files)
    ```

2.  **Add your credentials to `secrets.toml`:**
    Open the newly created `secrets.toml` file and add your keys and configuration details using the following format. **Replace the placeholder values with your actual credentials.**

    ```toml
    # .streamlit/secrets.toml

    OPENAI_API_KEY = "sk-YOUR_OPENAI_API_KEY"
    LLAMA_CLOUD_API_KEY = "llx-YOUR_LLAMA_CLOUD_API_KEY"

    # These LlamaCloud details are crucial for connecting to your specific index.
    # The application uses default values ("Default" for project name, "vishal-pdf-parsing" for index name)
    # if these are not specified, but it's best to define them explicitly.
    LLAMA_CLOUD_PROJECT_NAME = "YourLlamaCloudProjectName" # e.g., "Default"
    LLAMA_CLOUD_INDEX_NAME = "YourLlamaCloudIndexName"   # e.g., "vishal-pdf-parsing"
    LLAMA_CLOUD_ORGANIZATION_ID = "YourLlamaCloudOrgID"  # Your LlamaCloud Organization ID
    ```

    **Important Notes:**
    -   The `app.py` file includes default values for `LLAMA_CLOUD_PROJECT_NAME` (`"Default"`) and `LLAMA_CLOUD_INDEX_NAME` (`"vishal-pdf-parsing"`) if they are omitted from `secrets.toml`. However, `OPENAI_API_KEY`, `LLAMA_CLOUD_API_KEY`, and `LLAMA_CLOUD_ORGANIZATION_ID` are mandatory.
    -   The `.streamlit/secrets.toml` file is included in the `.gitignore` file, ensuring that your sensitive API keys are not accidentally committed to version control.

## 🚀 Usage

Once the installation and configuration steps are complete, you can run the application:

1.  **Ensure your virtual environment is activated** (if you created one).
2.  **Navigate to the project's root directory** in your terminal.
3.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
4.  The application will automatically open in your default web browser (usually at `http://localhost:8501`).
5.  You can now interact with "Professor Gupta" by typing your questions related to the Leadership Skills course in the chat input field at the bottom of the page. The AI will provide answers based on the course materials.

## 🤝 Contributing (Optional)

Contributions are welcome! If you'd like to improve the chatbot or add new features, please consider the following:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix (`git checkout -b feature/your-exciting-feature`).
3.  Make your changes and commit them with clear messages (`git commit -am 'Add some exciting feature'`).
4.  Push your changes to your forked repository (`git push origin feature/your-exciting-feature`).
5.  Open a Pull Request to the main repository.

## 📄 License (Optional)

This project can be licensed under your choice of open-source license. If you add a `LICENSE` file, mention it here (e.g., "This project is licensed under the MIT License - see the `LICENSE` file for details.").

---

Happy learning with Professor Gupta!
