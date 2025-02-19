const express = require('express');
const multer = require('multer');
const csv = require('csv-parser');
const fs = require('fs');
const path = require('path');
const cors = require('cors');

const app = express();
const port = 3001; // Change to 3001

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// Multer setup for file uploads
const upload = multer({ dest: 'uploads/' });

// In-memory storage for dataset
let dataset = [];

// Route to handle file upload
app.post('/upload', upload.single('file'), (req, res) => {
    if (!req.file) {
        return res.status(400).json({ message: 'No file uploaded' });
    }

    const filePath = req.file.path;
    dataset = []; // Clear previous data

    fs.createReadStream(filePath)
        .pipe(csv())
        .on('data', (row) => {
            dataset.push(row);
        })
        .on('end', () => {
            fs.unlinkSync(filePath); // Delete the uploaded file after processing
            res.json({ message: 'File uploaded and processed successfully', data: dataset });
        })
        .on('error', (err) => {
            console.error('Error processing CSV file:', err);
            res.status(500).json({ message: 'Error processing CSV file' });
        });
});

// Route to get dataset
app.get('/data', (req, res) => {
    res.json(dataset);
});

// Route to flag a row
app.post('/flag', (req, res) => {
    const { id } = req.body;
    const row = dataset.find((item) => item.ID === id);
    if (row) {
        row.Flagged = true;
        res.json({ message: 'Row flagged successfully', row });
    } else {
        res.status(404).json({ message: 'Row not found' });
    }
});

// Route to export CSV
app.get('/export', (req, res) => {
    const csvData = dataset.map((row) => Object.values(row).join(',')).join('\n');
    res.setHeader('Content-Type', 'text/csv');
    res.setHeader('Content-Disposition', 'attachment; filename=exported_data.csv');
    res.send(csvData);
});

// Serve the frontend
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Start the server
app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});