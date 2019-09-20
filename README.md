# How to start the app locally

Create virtual environment. From **this** directory:

`python -m venv reco_app`

`source reco_app/bin/activate`

`cd reco_app`

Install required packages:

`pip install -r requirements.txt`

Set environmental variables and run:

`export FLASK_ENV=development`

`export FLASK_APP=app.py`

`flask run --no-reload`

The application can now be accessed by default from: localhost:5000

# How to start the app with Docker

- Install docker and docker-compose.
- From the base directory run `docker-compose up`
- App will be available from localhost:5000


# How to use the service

- Enter the User ID. If no User ID is provided, or it does not exists in the dataset, then the recommendations are computed only based on mean ratings and genre/context biases. Otherwise, factorized matrices are taken into consideration too.
- Select Context Category from the list. Then select one of its possible values from the second list.
- Click **Get Recommendations** button to see the recommendations.
- You can view the current model's quality metrics by clicking on *See model's quality metrics*.

When the server is started, Matrix Factorization algorithm can be run on the dataset for a specified number of epochs (it then takes some time before the service is ready to use for the first time). The resulting matrices are then stored in pickles for quick access after restart. Pre-computed matrices are provided and loaded by the app by default.

By default the app returns 10 tracks with highest predicted ratings. In a real system these tracks should be shuffled and the user's history should be stored in order not to recommend always the same tracks.

