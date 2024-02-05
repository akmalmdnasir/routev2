from flask import Flask, render_template, request, send_file
import os
import pandas as pd
from main import process_dataframe  # Import the process_dataframe function


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the POST request has the file part
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')

        file = request.files['file']

        # If the user does not select a file, the browser submits an empty file
        if file.filename == '':
            return render_template('index.html', message='No selected file')
        
        folder_path = "uploads"

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Folder '{folder_path}' created successfully.")

        # Save the file to the uploads folder
        file_path = os.path.join(folder_path, file.filename)
        file.save(file_path)

        # Process the uploaded CSV file
        df = pd.read_csv(file_path)
        result_file_path = process_dataframe(df)

        return render_template('index.html', message='File uploaded and processed successfully', result_file=result_file_path)
    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
