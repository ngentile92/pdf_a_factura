import os
import sys

from flask import (Flask, flash, redirect, render_template, request, session,
                   url_for)
from werkzeug.utils import secure_filename

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Your existing Python code for processing the PDF
from pdf_to_csv import process_pdf

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'your_secret_key'  # Replace with your own secret key

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_pdf():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        # If the user does not select a file, the browser submits an empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Process the PDF and get the relevant information
            pdf_info = process_pdf(file_path)

            # Store the processed PDF data in the session
            session['pdf_info'] = pdf_info

            # Redirect the user to the results page
            return redirect(url_for('display_results'))

    return render_template('upload.html')

@app.route('/results', methods=['GET'])
def display_results():
    # Get the processed PDF data from the session
    pdf_info = session.get('pdf_info', None)
    if pdf_info is None:
        # If the user tries to access the results page directly without uploading a PDF, redirect them to the upload page
        flash('Please upload a PDF first.')
        return redirect(url_for('upload_pdf'))
    else:
        # Render the results page with the PDF data
        return render_template('results.html', pdf_info=pdf_info)



if __name__ == '__main__':
    app.run(debug=True)
