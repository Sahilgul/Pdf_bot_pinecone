import os
from dotenv import load_dotenv
import fitz
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from flask import Flask, render_template, request, jsonify
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI

load_dotenv()

# Access the environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = "gcp-starter"
             

pinecone.init(      
	api_key=PINECONE_API_KEY,      
	environment='gcp-starter'      
) 
index_name = pinecone.Index('pdfbot')     
#index = pinecone.Index('pdfbot')

active_indexes = pinecone.list_indexes()
print(active_indexes)

index_description = pinecone.describe_index("pdfbot")
print(index_description)

index = pinecone.Index("pdfbot")
index_stats_response = index.describe_index_stats()
print(index_stats_response)

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Define PDF file path#
#pdf_path = "C:/Users/NKU/Downloads/Chatbot_with_pdf/2nation_theory.pdf."

#  #Extract text from PDF file
# with open(pdf_path, "rb") as f:
#      pdf_content = f.read()

# # # Use PyMuPDF to extract text from each page
# pdf_document = fitz.open("pdf", pdf_content)
# book_texts = [page.get_text() for page in pdf_document]
# # pdf_document.close()

# # # Remove duplicates from the extracted text
# unique_texts = list(set(book_texts))

# # Create vectors using Pinecone
# # book_docsearch = Pinecone.from_texts([t.page_content for t in book_texts], embeddings, index_name = index_name)
# book_docsearch = Pinecone.from_texts(unique_texts, embeddings, index_name="pdfbot")

# def fetch_vectors(ids_to_fetch):
#     query_response = book_docsearch.search(
#         search_type="full_text",
#         query="",
#         ids=ids_to_fetch
#     )
# Create Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return 'pdfbot'

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    global book_docsearch

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    pdf_content = file.read()
    pdf_document = fitz.open("pdf", pdf_content)
    book_texts = [page.get_text() for page in pdf_document]
    pdf_document.close()

    unique_texts = list(set(book_texts))

    # Create vectors using Pinecone
    book_docsearch = Pinecone.from_texts(unique_texts, embeddings, index_name="pdfbot")

    return jsonify({'success': True, 'message': 'PDF uploaded successfully'})



@app.route("/chatbot", methods=["POST"])
def chatbot_query():
    # Extract query from request data
    query = request.json["query"]

    # Perform similarity search using Pinecone
    docs = book_docsearch.similarity_search(query)

    # Load QA chain and run it with the retrieved documents
    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    chain = load_qa_chain(llm, chain_type="stuff")
    answer = chain.run(input_documents=docs, question=query)

    # Return the answer in JSON format
    return jsonify({"answer": answer})

# /delete_index route
@app.route("/delete_index", methods=["DELETE"])
def delete_index():
    try:
        # Extract the index name from the request data
        index_name = request.json.get("index_name")

        if not index_name:
            return jsonify({"error": "Index name is required"}), 400

        # Check if the index exists
        if index_name not in pinecone.list_indexes():
            return jsonify({"error": f"Index '{index_name}' does not exist"}), 404

        # Delete the specified index in Pinecone
        pinecone.delete_index(index_name)
        return jsonify({"success": f"Index '{index_name}' deleted successfully"})
    except Exception as e:
        return jsonify({"error": f"Failed to delete index '{index_name}': {str(e)}"}), 500

@app.route("/delete", methods=["POST"])
def delete_vectors():
    # Extract document IDs to delete from request data
    ids_to_delete = request.json["ids"]

    # Delete vectors in Pinecone index
    delete_response = book_docsearch.delete(ids=ids_to_delete)

    # Return the delete response in JSON format
    return jsonify({"delete_response": delete_response})
    
@app.route("/fetch_vectors", methods=["POST"])
def fetch_vectors():
    try:
        # Extract document IDs to fetch from request data
        ids_to_fetch = request.json["ids"]

        # Fetch vectors from Pinecone index
        fetch_response = index.fetch(ids=ids_to_fetch)
        # print(fetch_response)

        # Check if fetch response is successful and data is available
        if fetch_response:
            # Fetching text from metadata
            text = fetch_response['vectors'][ids_to_fetch[0]]['metadata']['text']

            # Printing the fetched text
            print(text)
            return jsonify(
                {
                    "text":text
                }
            )
        else:
            return jsonify({"error": "failed to fetch_response"}), 500

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return jsonify({"error": str(e)})

@app.route("/create_index", methods=["POST"])
def create_index():
    try:
        # Extract index name and dimension from request data
        index_name = request.json["index_name"]
        dimension = request.json["dimension"]

        # Create the index in Pinecone
        pinecone.create_index(index_name, dimension=dimension)

        return jsonify({"success": f"Index '{index_name}' created successfully"})
    except Exception as e:
        return jsonify({"error": f"Failed to create index '{index_name}': {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)