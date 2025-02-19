const express = require('express');
const multer = require('multer');
const csv = require('csv-parser');
const fs = require('fs');
const path = require('path');
const cors = require('cors');
const axios = require('axios');

const app = express();
const port = process.env.PORT || 3001; // Use environment variable or default to 3001

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// Multer setup for file uploads
const upload = multer({ dest: 'uploads/' });

// In-memory storage for dataset
let dataset = [];

// IBM Watson API Key
const API_KEY = ""; // Replace with your actual API key

// Function to get IBM Watson authentication token
async function getAuthToken() {
    try {
        const response = await axios.post(
            'https://iam.cloud.ibm.com/identity/token',
            null,
            {
                params: {
                    apikey: API_KEY,
                    grant_type: 'urn:ibm:params:oauth:grant-type:apikey',
                },
            }
        );
        return response.data.access_token;
    } catch (error) {
        console.error('Error getting auth token:', error);
        throw new Error('Failed to retrieve token');
    }
}

// Function to get predictions from IBM Watson
async function getPrediction(transactions) {
    try {
        const token = await getAuthToken();
        const payload = { input_data: transactions.input_data };

        const response = await axios.post(
            'https://eu-de.ml.cloud.ibm.com/ml/v4/deployments/testing1/predictions?version=2021-05-01',
            payload,
            {
                headers: {
                    Authorization: `Bearer ${token}`,
                    'Content-Type': 'application/json',
                },
            }
        );

        return response.data.predictions;
    } catch (error) {
        console.error('Error during prediction:', error);
        throw new Error('Failed to make prediction');
    }
}

// Endpoint to handle Watson AI predictions
app.post('/predict', async (req, res) => {
    try {
        const transactions = req.body.transactions;
        const predictions = await getPrediction(transactions);
        res.json({ predictions });
    } catch (error) {
        res.status(500).json({ error: 'Error processing prediction' });
    }
});

// Endpoint to handle CSV file uploads
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

// Endpoint to get processed dataset
app.get('/data', (req, res) => {
    res.json(dataset);
});

// Endpoint to flag a specific row
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

// Endpoint to export dataset as CSV
app.get('/export', (req, res) => {
    const csvData = dataset.map((row) => Object.values(row).join(',')).join('\n');
    res.setHeader('Content-Type', 'text/csv');
    res.setHeader('Content-Disposition', 'attachment; filename=exported_data.csv');
    res.send(csvData);
});

// Serve frontend
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Start the server
app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});
