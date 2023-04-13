import os
import streamlit as st
from pdf_to_csv import process_pdf
from database import main as pdf_db
ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def main():
    st.set_page_config(page_title='PDF Processing App')
    st.title('PDF Processing App')

    os.makedirs('./uploads', exist_ok=True)

    uploaded_files = st.file_uploader('Upload one or more PDF files', type=ALLOWED_EXTENSIONS, accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if allowed_file(uploaded_file.name):
                file_path = os.path.join('./uploads', uploaded_file.name)
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())

                # Process the PDF and get the relevant information
                folder_path = os.path.dirname(file_path)
                pdf_info = process_pdf(folder_path)

                # Store the processed PDF data in the database
                pdf_db().insert_data(pdf_info)
                # Render the results page with the PDF data
                st.subheader(f'PDF Information for {uploaded_file.name}')
                st.write(pdf_info)
            else:
                st.error(f'Please upload a PDF file: {uploaded_file.name}')

if __name__ == '__main__':
    main()
