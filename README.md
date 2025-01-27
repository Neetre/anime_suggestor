# anime_suggestor

## Description

This is a simple anime suggestor that suggests anime based on the genre you like, created using Streamlit (it's a little heavy to load, so please be patient).
The suggestor uses a statistical model to predict the anime that you might like based on the genre you like.
The dataset used is from the [MyAnimeList dataset](https://www.kaggle.com/azathoth42/myanimelist).
The dataset contains information about anime such as the name, genre, rating, etc.

### Note

The dataset is not included in the repository. You can download it from the link above.
Also, for now the suggestor only suggests anime based on the dataset. In the future, I plan to add more features to the suggestor, like suggesting anime based on the user's rating, etc. So I will be adding a login system to the suggestor, in order to store the user's rating.

![anime_suggestor](https://cloud-cv7ae9o04-hack-club-bot.vercel.app/0image.png)

## Requirements

- Python 3.9 or higher

### Python Libraries

1. Create a virtual environment

    ```bash
    python -m venv .venv
    ```

2. Activate the virtual environment

    ```bash
    source .venv/bin/activate
    ```

3. Install the required libraries

    ```bash
    pip install -r requirements.txt
    ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

[Neetre](https://github.com/Neetre)
