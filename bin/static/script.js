BASE_URL = 'http://an.figliolo.it';

document.addEventListener('DOMContentLoaded', function () {
    const pageSelector = document.getElementById('page-selector');
    const pages = document.querySelectorAll('.page');

    function fetchAndDisplayGenres() {
        fetch(BASE_URL + '/genres')
            .then(response => response.json())
            .then(data => {
                const genresList = document.getElementById('genres-list');
                genresList.textContent = data.slice(0, 20).join(', ') + '...';
            })
            .catch(error => {
                console.error('Error fetching genres:', error);
            });
    }

    function fetchAndDisplayTypes() {
        fetch(BASE_URL + '/types')
            .then(response => response.json())
            .then(data => {
                const typesList = document.getElementById('types-list');
                typesList.textContent = data.join(', ');
            })
            .catch(error => {
                console.error('Error fetching types:', error);
            });
    }

    fetchAndDisplayGenres();
    fetchAndDisplayTypes();

    pageSelector.addEventListener('change', function () {
        pages.forEach(page => page.style.display = 'none');
        const selectedPage = document.getElementById(this.value);
        if (selectedPage) {
            selectedPage.style.display = 'block';
        }
    });

    fetch(BASE_URL + '/starter-anime')
        .then(response => response.json())
        .then(data => {
            const starterAnimeList = document.getElementById('starter-anime-list');
            starterAnimeList.innerHTML = data.map(anime => `
                <div class="expander">
                    <h3>${anime.name} (${anime.type})</h3>
                    <div class="expander-content">
                        <p><strong>Genre:</strong> ${anime.genre}</p>
                        <p><strong>Rating:</strong> ${anime.rating}</p>
                        <p><strong>Episodes:</strong> ${anime.episodes}</p>
                    </div>
                </div>
            `).join('');
        })
        .catch(error => {
            console.error('Error fetching starter anime:', error);
        });

    // Populate genres with checkboxes
    fetch(BASE_URL + '/genres')
        .then(response => response.json())
        .then(data => {
            const favoriteGenres = document.getElementById('favorite-genres');
            favoriteGenres.innerHTML = data.map(genre => `
                <label class="checkbox-label">
                    <input type="checkbox" value="${genre}"> ${genre}
                </label>
            `).join('');
        })
        .catch(error => {
            console.error('Error populating favorite genres:', error);
        });

    // Populate types with checkboxes
    fetch(BASE_URL + '/types')
        .then(response => response.json())
        .then(data => {
            const preferredTypes = document.getElementById('preferred-types');
            preferredTypes.innerHTML = data.map(type => `
                <label class="checkbox-label">
                    <input type="checkbox" value="${type}"> ${type}
                </label>
            `).join('');
        })
        .catch(error => {
            console.error('Error populating preferred types:', error);
        });

    const maxEpisodes = document.getElementById('max-episodes');
    const maxEpisodesValue = document.getElementById('max-episodes-value');
    maxEpisodes.addEventListener('input', function () {
        maxEpisodesValue.textContent = this.value;
    });

    const minRating = document.getElementById('min-rating');
    const minRatingValue = document.getElementById('min-rating-value');
    minRating.addEventListener('input', function () {
        minRatingValue.textContent = this.value;
    });

    const getRecommendationsButton = document.getElementById('get-recommendations');
    getRecommendationsButton.addEventListener('click', function () {
        const favoriteGenres = Array.from(document.querySelectorAll('#favorite-genres input[type="checkbox"]:checked')).map(checkbox => checkbox.value);
        const preferredTypes = Array.from(document.querySelectorAll('#preferred-types input[type="checkbox"]:checked')).map(checkbox => checkbox.value);
        const maxEpisodes = document.getElementById('max-episodes').value;
        const minRating = document.getElementById('min-rating').value;

        const preferences = {
            favorite_genres: favoriteGenres,
            preferred_types: preferredTypes,
            max_episodes: parseInt(maxEpisodes),
            min_rating: parseFloat(minRating)
        };

        fetch(BASE_URL + '/initial-recommendations', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(preferences)
        })
        .then(response => response.json())
        .then(data => {
            const recommendationsList = document.getElementById('recommendations-list');
            recommendationsList.innerHTML = data.map(anime => `
                <div class="expander">
                    <h3>${anime.name} (${anime.type})</h3>
                    <div class="expander-content">
                        <p><strong>Genre:</strong> ${anime.genre}</p>
                        <p><strong>Rating:</strong> ${anime.rating}</p>
                        <p><strong>Episodes:</strong> ${anime.episodes}</p>
                    </div>
                </div>
            `).join('');
        })
        .catch(error => {
            console.error('Error fetching recommendations:', error);
        });
    });
});