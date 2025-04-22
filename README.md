# LeetRevise
LeetRevise – Personalized LeetCode Reccommendation &amp; Smart Revision System
# LeetCode Question Recommender

A full-stack web application that provides personalized LeetCode question recommendations based on your solved problems. The system analyzes your LeetCode profile, identifies patterns in your solved questions, and recommends new problems that will help you improve your skills.

## Demo Video

<video width="100%" controls>
  <source src="https://raw.githubusercontent.com/qusaii21/LeetRevise/main/assets/demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## Features

- **User Profile Analysis**: Connect with your LeetCode username to analyze your solved problems
- **Dual Recommendation Methods**: Get recommendations based on either question descriptions or solution patterns
- **AI-Powered Curation**: Uses Gemini AI to curate the top 5 most relevant questions from a pool of 15 recommendations
- **Interactive UI**: Clean, responsive interface with detailed problem information
- **Direct LeetCode Links**: Easily access recommended problems on LeetCode

## Tech Stack

### Backend
- **Flask**: Python web framework for the API
- **scikit-learn**: For TF-IDF vectorization, LDA topic modeling, and nearest neighbors analysis
- **NetworkX**: For graph-based recommendation algorithms
- **pandas**: For data manipulation and analysis
- **Google Generative AI**: For analyzing and curating recommendations
- **Hugging Face Datasets**: For accessing the LeetCode dataset

### Frontend
- **React.js**: JavaScript library for building the user interface
- **Axios**: For making API requests to the backend
- **CSS3**: For styling the application

### External Dependencies
- **[alfa-leetcode-api](https://github.com/alfaarghya/alfa-leetcode-api)**: Used for accessing LeetCode user data and submissions

## Getting Started

### Prerequisites
- Python 3.7+
- Node.js 14+
- npm 6+
- Google Gemini API key
- alfa-leetcode-api running locally

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/leetcode-recommender.git
cd leetcode-recommender
```

2. Set up the LeetCode API
```bash
# Clone the alfa-leetcode-api repository
git clone https://github.com/alfaarghya/alfa-leetcode-api.git
cd alfa-leetcode-api

# Follow the setup instructions in the alfa-leetcode-api README
# Start the API server (typically runs on port 3000)
```

3. Set up the backend
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export FLASK_APP=app.py
export FLASK_ENV=development
export GEMINI_API_KEY=your_gemini_api_key_here
```

4. Set up the frontend
```bash
cd frontend
npm install
```

### Running the Application

1. Ensure the alfa-leetcode-api is running on http://localhost:3000

2. Start the backend server
```bash
# From the root directory
python app.py
```

3. Start the frontend development server
```bash
# From the frontend directory
npm start
```

4. Open your browser and navigate to `http://localhost:3000`

## Usage

1. Enter your LeetCode username in the input field
2. Select the recommendation type (based on descriptions or solutions)
3. Click "Get Recommendations"
4. View the full list of 15 recommendations and the AI-curated top 5 picks
5. Click on the LeetCode links to try the recommended problems

## API Endpoints

### `GET /api/health`
- Health check endpoint

### `GET /api/user/<username>/solved`
- Fetches all solved problems for a specific user
- Returns a list of problem slugs

### `POST /api/recommendations`
- Generates recommendations based on solved problems
- Request body:
  ```json
  {
    "username": "leetcode_username",
    "recommendation_type": "description" // or "solution"
  }
  ```
- Returns recommendations and AI-curated top picks

## Project Structure

```
leetcode-recommender/
├── app.py                  # Flask application
├── requirements.txt        # Python dependencies
├── updated_data.json       # Question dataset
├── README.md               # This file
└── frontend/              
    ├── package.json        # npm dependencies
    ├── public/             # Static files
    └── src/                # React source code
        ├── App.js          # Main React component
        ├── App.css         # Main styles
        └── components/     # UI components
```

## Generating the Demo Video

To create a demo video for your project:

1. Record a screen capture demonstrating:
   - Installation process
   - Starting the application
   - Entering a username
   - Selecting recommendation methods
   - Viewing and exploring recommendations
   - Highlighting key features

2. Edit the video to be concise (2-3 minutes) while showing all major functionality
3. Upload to YouTube or another video hosting platform
4. Update the link in this README

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- LeetCode platform for the problem dataset
- [alfa-leetcode-api](https://github.com/alfaarghya/alfa-leetcode-api) for providing LeetCode data access
- Hugging Face for providing the dataset access
- Google for the Gemini AI API
